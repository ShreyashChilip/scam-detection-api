import os
import joblib
import re
import string
import assemblyai as aai
import json
import time
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Load ML model and vectorizer
model = joblib.load("models/logistic_regression_model_new.pkl")
vectorizer = joblib.load("models/vectorizer_new.pkl")

def preprocess_text(text):
    """Preprocess text consistently with training data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_feature_importance(text, vectorizer, model, top_n=5):
    """Extract feature importance for prediction explanation."""
    try:
        # Get feature names and model weights
        feature_names = vectorizer.get_feature_names_out()
        weights = model.coef_[0]
        
        # Transform text and get contributions
        X = vectorizer.transform([text])
        x = X.toarray()[0]
        contrib = x * weights
        
        # Get non-zero contributions
        explanations = []
        for i, v in enumerate(contrib):
            if v != 0:
                explanations.append({
                    "word": feature_names[i],
                    "contribution": round(float(v), 4),
                    "impact": "scam" if v > 0 else "safe"
                })
        
        # Sort by absolute impact
        explanations = sorted(explanations, key=lambda x: abs(x["contribution"]), reverse=True)
        
        return explanations[:top_n]
    except Exception as e:
        return [{"error": f"Feature analysis failed: {str(e)}"}]

# Educational response schema (for Flutter app parsing):
# {
#   "scam_type": "string",
#   "summary": "string",
#   "how_it_works": "string",
#   "prevention_tips": ["string"],
#   "red_flags": ["string"],
#   "what_to_do_if_targeted": "string"
# }

EDUCATION_JSON_SCHEMA = """{
  "scam_type": "short label for the scam type",
  "summary": "2-3 sentences on what this scam could be about",
  "how_it_works": "brief explanation of how this scam typically works",
  "prevention_tips": ["tip1", "tip2", "tip3"],
  "red_flags": ["warning sign 1", "warning sign 2"],
  "what_to_do_if_targeted": "actionable advice if user was targeted"
}"""


def get_scam_education_from_groq(detected_keywords, context_snippets):
    """Call Groq LLM to generate educational JSON about the detected scam patterns."""
    if not GROQ_AVAILABLE:
        return None, "Groq SDK not installed. pip install groq"
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, "GROQ_API_KEY environment variable not set"
    keywords_str = ", ".join(detected_keywords) if isinstance(detected_keywords, list) else str(detected_keywords)
    context_str = "\n".join(context_snippets) if isinstance(context_snippets, list) else str(context_snippets)
    prompt = f"""You are a scam awareness expert. Based on the following scam-related keywords and the surrounding context from a detected call or message, generate educational content in the exact JSON format below.

Detected scam keywords: {keywords_str}

Context (neighboring words/sentences where these appeared):
{context_str}

Return ONLY a single valid JSON object (no markdown, no code block, no extra text) with this exact structure. Use double quotes for keys and strings. Escape any quotes inside strings.
{EDUCATION_JSON_SCHEMA}

Fill every field. prevention_tips and red_flags must be arrays of strings (3-5 items each). Be concise and actionable."""
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        raw = completion.choices[0].message.content.strip()
        # Strip markdown code block if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```\s*$", "", raw)
        data = json.loads(raw)
        # Normalize to expected keys
        expected_keys = {"scam_type", "summary", "how_it_works", "prevention_tips", "red_flags", "what_to_do_if_targeted"}
        for k in expected_keys:
            if k not in data:
                data[k] = "" if k in ("scam_type", "summary", "how_it_works", "what_to_do_if_targeted") else []
        return data, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON from LLM: {e}"
    except Exception as e:
        return None, str(e)


def get_warning_message(probability):
    """Generate a warning message based on confidence level."""
    if probability >= 0.90:
        return "☠️ This call is highly suspicious! Disconnect immediately."
    elif probability >= 0.75:
        return "🚨 Strong scam indicators detected! Be cautious."
    elif probability >= 0.60:
        return "⚠️ Some scam-like patterns detected."
    elif probability >= 0.50:
        return "🚧 Slightly below scam threshold, verify details."
    else:
        return "✅ No scam detected, but stay cautious."

@app.route('/predict-audio-stream', methods=['POST'])
def predict_audio_stream():
    """API endpoint for streaming audio prediction with real-time scam detection.
    
    This endpoint transcribes audio using AssemblyAI and analyzes each utterance
    in real-time. If any utterance exceeds the scam threshold, it immediately
    returns an alert and stops processing.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        file_path = "temp_audio_stream.wav"
        audio_file.save(file_path)

        # Use AssemblyAI with speaker labels for transcription
        config = aai.TranscriptionConfig(speaker_labels=True)
        transcript = aai.Transcriber().transcribe(file_path, config)

        if not transcript.utterances:
            return jsonify({
                "final_prediction": "NOT SCAM",
                "final_confidence": 0.0,
                "total_utterances": 0,
                "message": "No speech detected in audio",
                "utterance_results": [],
                "status": "completed"
            })

        # Process utterances as streaming chunks
        accumulated_texts = []
        accumulated_confidences = []
        utterance_results = []
        speaker_scam_flags = {}  # Track which speakers have triggered scam alerts

        for utterance_count, utterance in enumerate(transcript.utterances, 1):
            speaker = f"Speaker {utterance.speaker}"
            text_chunk = utterance.text.strip()

            # Process current utterance
            processed_text = preprocess_text(text_chunk)

            if not processed_text:
                utterance_result = {
                    "utterance": utterance_count,
                    "speaker": speaker,
                    "text": text_chunk[:100] + "..." if len(text_chunk) > 100 else text_chunk,
                    "prediction": "NOT SCAM",
                    "confidence": 0.0,
                    "message": "No valid text to analyze",
                    "status": "processed"
                }
            else:
                # Predict for current utterance
                text_vectorized = vectorizer.transform([processed_text])
                probability = model.predict_proba(text_vectorized)[0][1]
                prediction = "SCAM" if probability >= 0.60 else "NOT SCAM"
                message = get_warning_message(probability)

                # Get feature importance for explanation
                feature_importance = get_feature_importance(processed_text, vectorizer, model)

                # Store for aggregation
                accumulated_texts.append(text_chunk)
                accumulated_confidences.append(probability)

                utterance_result = {
                    "utterance": utterance_count,
                    "speaker": speaker,
                    "text": text_chunk[:100] + "..." if len(text_chunk) > 100 else text_chunk,
                    "prediction": prediction,
                    "confidence": round(probability, 2),
                    "message": message,
                    "feature_importance": feature_importance,
                    "status": "processed"
                }

                # IMMEDIATE SCAM ALERT - if threshold exceeded, signal immediately
                if probability >= 0.60:
                    speaker_scam_flags[speaker] = True
                    utterance_results.append(utterance_result)

                    try:
                        # Educational content: scam keywords + context -> Groq
                        print(f"DEBUG: Processing scam alert. Feature importance: {feature_importance}")
                        detected_keywords = [
                            x["word"] for x in feature_importance
                            if x.get("impact") == "scam"
                        ]
                        context_snippets = [text_chunk] if text_chunk else []
                        print(f"DEBUG: Calling Groq with keywords={detected_keywords}")
                        print(f"DEBUG: Groq context_snippets={context_snippets}")
                        
                        education, education_error = get_scam_education_from_groq(
                            detected_keywords or ["suspicious language"],
                            context_snippets or ["No context available."],
                        )
                        print(f"DEBUG: Groq result: education={json.dumps(education, indent=2) if education else 'None'}, error={education_error}")
                        
                        response_payload = {
                            "alert": "SCAM DETECTED",
                            "scam_detected_at_utterance": utterance_count,
                            "scam_speaker": speaker,
                            "scam_confidence": round(probability, 2),
                            "message": f"🚨 SCAM DETECTED from {speaker}! Analysis stopped immediately.",
                            "final_prediction": "SCAM",
                            "final_confidence": round(probability, 2),
                            "utterance_results": utterance_results,
                            "educational": education,
                            "status": "scam_detected",
                        }
                        if education_error and education is None:
                            response_payload["educational_error"] = education_error

                        # Clean up before returning
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                            except:
                                pass

                        return jsonify(response_payload)
                        
                    except Exception as e:
                        print(f"ERROR in predict-audio-stream scam block: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Fallback response if something fails in education generation
                        return jsonify({
                            "alert": "SCAM DETECTED",
                            "error": f"Internal processing error: {str(e)}",
                            "final_prediction": "SCAM",
                            "chunk_results": utterance_results
                        }), 500

            utterance_results.append(utterance_result)

        # Final aggregated result (only if no scam was detected)
        if accumulated_confidences:
            final_confidence = sum(accumulated_confidences) / len(accumulated_confidences)
            final_prediction = "NOT SCAM"  # We know it's not scam since we didn't exit early

            return jsonify({
                "final_prediction": final_prediction,
                "final_confidence": round(final_confidence, 2),
                "total_utterances": len(transcript.utterances),
                "message": get_warning_message(final_confidence),
                "utterance_results": utterance_results,
                "status": "completed"
            })
        else:
            return jsonify({
                "final_prediction": "NOT SCAM",
                "final_confidence": 0.0,
                "total_utterances": len(transcript.utterances),
                "message": "No valid text analyzed",
                "utterance_results": utterance_results,
                "status": "completed"
            })

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

@app.route('/predict-text-stream', methods=['POST'])
def predict_text_stream():
    """API endpoint for streaming text prediction with real-time scam detection."""
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({"error": "No texts provided"}), 400
        
        texts = data['texts']  # Expecting a list of text chunks
        
        accumulated_texts = []
        accumulated_confidences = []
        chunk_results = []
        
        for chunk_count, text_chunk in enumerate(texts, 1):
            # Process current chunk
            processed_text = preprocess_text(text_chunk)
            
            if not processed_text:
                chunk_result = {
                    "chunk": chunk_count,
                    "text": text_chunk[:100] + "..." if len(text_chunk) > 100 else text_chunk,
                    "prediction": "NOT SCAM",
                    "confidence": 0.0,
                    "message": "No valid text to analyze",
                    "status": "processed"
                }
            else:
                # Predict for current chunk
                text_vectorized = vectorizer.transform([processed_text])
                probability = model.predict_proba(text_vectorized)[0][1]
                prediction = "SCAM" if probability >= 0.60 else "NOT SCAM"
                message = get_warning_message(probability)
                
                # Get feature importance for explanation
                feature_importance = get_feature_importance(processed_text, vectorizer, model)
                
                # Store for aggregation
                accumulated_texts.append(text_chunk)
                accumulated_confidences.append(probability)
                
                chunk_result = {
                    "chunk": chunk_count,
                    "text": text_chunk[:100] + "..." if len(text_chunk) > 100 else text_chunk,
                    "prediction": prediction,
                    "confidence": round(probability, 2),
                    "message": message,
                    "feature_importance": feature_importance,
                    "status": "processed"
                }
                
                # IMMEDIATE SCAM ALERT - if threshold exceeded, signal immediately
                if probability >= 0.60:
                    chunk_results.append(chunk_result)
                    
                    # Educational content: scam keywords + context -> Groq
                    detected_keywords = [
                        x["word"] for x in feature_importance
                        if x.get("impact") == "scam"
                    ]
                    context_snippets = [text_chunk] if text_chunk else []
                    education, education_error = get_scam_education_from_groq(
                        detected_keywords or ["suspicious language"],
                        context_snippets or ["No context available."],
                    )
                    
                    response_payload = {
                        "alert": "SCAM DETECTED",
                        "scam_detected_at_chunk": chunk_count,
                        "scam_confidence": round(probability, 2),
                        "message": "🚨 SCAM DETECTED! Analysis stopped immediately.",
                        "final_prediction": "SCAM",
                        "final_confidence": round(probability, 2),
                        "chunk_results": chunk_results,
                        "educational": education,
                        "status": "scam_detected"
                    }
                    if education_error and education is None:
                        response_payload["educational_error"] = education_error
                    
                    return jsonify(response_payload)
            
            chunk_results.append(chunk_result)
        
        # Final aggregated result (only if no scam was detected)
        if accumulated_confidences:
            final_confidence = sum(accumulated_confidences) / len(accumulated_confidences)
            final_prediction = "NOT SCAM"  # We know it's not scam since we didn't exit early
            
            return jsonify({
                "final_prediction": final_prediction,
                "final_confidence": round(final_confidence, 2),
                "total_chunks": len(texts),
                "message": get_warning_message(final_confidence),
                "chunk_results": chunk_results,
                "status": "completed"
            })
        else:
            return jsonify({
                "final_prediction": "NOT SCAM",
                "final_confidence": 0.0,
                "total_chunks": len(texts),
                "message": "No valid text analyzed",
                "chunk_results": chunk_results,
                "status": "completed"
            })
            
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/predict-text', methods=['POST'])
def predict_text():
    """API endpoint for single text prediction."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        processed_text = preprocess_text(text)
        
        if not processed_text:
            return jsonify({
                "prediction": "NOT SCAM",
                "confidence": "0.00",
                "message": "No valid text to analyze"
            })
        
        text_vectorized = vectorizer.transform([processed_text])
        probability = model.predict_proba(text_vectorized)[0][1]
        prediction = "SCAM" if probability >= 0.60 else "NOT SCAM"
        message = get_warning_message(probability)
        
        # Get feature importance for explanation
        feature_importance = get_feature_importance(processed_text, vectorizer, model)
        
        return jsonify({
            "prediction": prediction,
            "confidence": f"{probability:.2f}",
            "message": message,
            "feature_importance": feature_importance
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    """API endpoint to predict scam probability from an audio file."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        file_path = "temp_audio.wav"
        audio_file.save(file_path)

        config = aai.TranscriptionConfig(speaker_labels=True)
        transcript = aai.Transcriber().transcribe(file_path, config)

        speaker_texts = {}
        for utterance in transcript.utterances:
            speaker = f"Speaker {utterance.speaker}"
            text = utterance.text.strip()
            speaker_texts.setdefault(speaker, []).append(text)

        speaker_text_strings = {speaker: ". ".join(texts) + "." for speaker, texts in speaker_texts.items()}

        scam_confidences = []
        results = {}
        for speaker, text in speaker_text_strings.items():
            processed_text = preprocess_text(text)
            if not processed_text:
                results[speaker] = {
                    "prediction": "NOT SCAM",
                    "confidence": "0.00",
                    "message": "No valid text"
                }
            else:
                text_vectorized = vectorizer.transform([processed_text])
                probability = model.predict_proba(text_vectorized)[0][1]
                prediction = "SCAM" if probability >= 0.60 else "NOT SCAM"
                message = get_warning_message(probability)
                
                # Get feature importance for explanation
                feature_importance = get_feature_importance(processed_text, vectorizer, model)
                
                results[speaker] = {
                    "prediction": prediction,
                    "confidence": f"{probability:.2f}",
                    "message": message,
                    "feature_importance": feature_importance
                }
                if prediction == "SCAM":
                    scam_confidences.append(probability)

        final_confidence = sum(scam_confidences) / len(scam_confidences) if scam_confidences else 0.0
        final_prediction = "SCAM" if scam_confidences else "NOT SCAM"

        response_data = {
            "final_prediction": final_prediction,
            "final_confidence": f"{final_confidence:.2f}",  # Convert to string with 2 decimal places
            "speaker_details": results
        }

        print(speaker_text_strings)  # For debugging
        print(response_data)  # For debugging

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)