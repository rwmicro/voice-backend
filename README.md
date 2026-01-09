# RAG Voice Backend

- Text-to-Speech (TTS) multi-providers
  - Kokoro TTS
  - Chatterbox TTS
  - F5-TTS (clonage vocal zero-shot)
- Speech-to-Text (STT) avec Whisper
- Support GPU/CPU
- Gestion de file GPU pour optimisation des ressources
- WebSocket pour streaming temps réel
- API REST avec FastAPI
- Containerisé avec Docker

## Architecture

```
voice/
├── api/                  # API FastAPI
│   ├── main.py          # Point d'entrée principal
│   └── voice_service_unified.py
├── tts/                 # Providers TTS
│   ├── kokoro/
│   ├── chatterbox/
│   └── f5_tts/
├── stt/                 # Service STT
└── utils/               # Utilitaires
config/                  # Configuration
└── settings.py
```

## Installation

### Démarrage rapide

**Avec GPU:**
```bash
docker-compose up voice-backend
```

**Sans GPU (CPU uniquement):**
```bash
docker-compose --profile cpu up voice-backend-cpu
```

**Mode développement:**
```bash
docker-compose --profile dev up voice-backend-dev
```

## Configuration

Variables d'environnement dans `.env`:

```env
# Server
HOST=0.0.0.0
PORT=8002
LOG_LEVEL=INFO

# Models
TTS_DEFAULT_PROVIDER=chatterbox
STT_MODEL=openai/whisper-base

# GPU Queue
ENABLE_GPU_QUEUE=true

# Cache
TRANSFORMERS_CACHE=/app/models/transformers
HF_HOME=/app/models/huggingface
```

## API Endpoints

### Base URL
```
http://localhost:8002
```

### Health Check

**GET /** ou **GET /api/voice/health**

Vérification de l'état du service.

**Réponse:**
```json
{
  "status": "healthy",
  "version": "3.0.0",
  "models": {
    "stt": {"loaded": true, "model": "openai/whisper-base"},
    "tts": {"loaded": true, "provider": "chatterbox"}
  }
}
```

### Text-to-Speech

**POST /api/voice/tts**

Synthèse vocale à partir de texte.

**Requête:**
```json
{
  "text": "Hello world",
  "language": "en",
  "provider": "kokoro",
  "voice_id": "af_bella",
  "speed": 1.0,
  "audio_prompt_path": "/path/to/prompt.wav",
  "exaggeration": 0.5,
  "temperature": 0.8,
  "cfg_weight": 0.5,
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "Reference text"
}
```

**Paramètres:**
- `text` (requis): Texte à synthétiser (1-5000 caractères)
- `language` (optionnel): Code langue ISO 639-1 (défaut: "en")
- `provider` (optionnel): "kokoro", "chatterbox" ou "f5-tts"
- `voice_id` (optionnel): ID de la voix
- `speed` (optionnel): Vitesse de parole (0.5-2.0, défaut: 1.0)

**Paramètres Chatterbox:**
- `audio_prompt_path`: Chemin vers l'audio de référence
- `exaggeration`: Expressivité (0.25-2.0, défaut: 0.5)
- `temperature`: Température (0.05-5.0, défaut: 0.8)
- `cfg_weight`: Poids CFG (0.2-1.0, défaut: 0.5)

**Paramètres F5-TTS:**
- `ref_audio`: Chemin audio de référence pour clonage
- `ref_text`: Texte de référence correspondant à l'audio

**Réponse:**
```json
{
  "audio_base64": "UklGRiQAAABXQVZF...",
  "language": "en",
  "provider": "kokoro"
}
```

**Exemple cURL:**
```bash
curl -X POST http://localhost:8002/api/voice/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "language": "en",
    "provider": "kokoro",
    "voice_id": "af_bella"
  }'
```

### Speech-to-Text

**POST /api/voice/stt**

Transcription audio vers texte.

**Requête:**
- Content-Type: multipart/form-data
- Body: fichier audio (WAV, MP3, etc.)

**Réponse:**
```json
{
  "text": "transcribed text",
  "language": "en",
  "time": 1.234
}
```

**Exemple cURL:**
```bash
curl -X POST http://localhost:8002/api/voice/stt \
  -F "file=@audio.wav"
```

### Gestion des Providers

**GET /api/voice/tts/providers**

Liste des providers TTS disponibles.

**Réponse:**
```json
{
  "providers": ["kokoro", "chatterbox", "f5-tts"],
  "current": "chatterbox",
  "available_voices": [
    {
      "id": "af_bella",
      "name": "Bella",
      "language": "en"
    }
  ]
}
```

**POST /api/voice/tts/switch**

Changer de provider TTS.

**Requête:**
```json
{
  "provider": "kokoro",
  "model_variant": "F5-TTS"
}
```

**Réponse:**
```json
{
  "success": true,
  "provider": "kokoro",
  "message": "Switched to kokoro"
}
```

### Voix Disponibles

**GET /api/voice/tts/voices**

Liste des voix disponibles pour le provider actuel.

**Paramètres de requête:**
- `language` (optionnel): Filtrer par langue

**Réponse:**
```json
{
  "voices": [
    {
      "id": "af_bella",
      "name": "Bella",
      "language": "en",
      "gender": "female"
    }
  ]
}
```

**Exemple:**
```bash
curl http://localhost:8002/api/voice/tts/voices?language=en
```

### Gestion GPU

**GET /api/voice/gpu/status**

Statut de la file GPU.

**Réponse:**
```json
{
  "queue_enabled": true,
  "queue_size": 2,
  "active_tasks": 1,
  "gpu_memory_used": "4.2 GB",
  "gpu_memory_total": "8.0 GB"
}
```

**POST /api/voice/unload**

Décharger les modèles TTS de la mémoire GPU.

**Réponse:**
```json
{
  "success": true,
  "message": "TTS models unloaded"
}
```

### WebSocket Streaming

**WS /api/voice/stream**

Connexion WebSocket pour streaming bidirectionnel.

**Messages entrants:**

STT:
```json
{
  "type": "stt",
  "audio": "base64_encoded_audio"
}
```

TTS:
```json
{
  "type": "tts",
  "text": "Hello world",
  "language": "en",
  "voice_id": "af_bella"
}
```

Ping:
```json
{
  "type": "ping"
}
```

**Messages sortants:**

Résultat STT:
```json
{
  "type": "stt_result",
  "text": "transcribed text",
  "language": "en",
  "time": 1.234
}
```

Chunk TTS:
```json
{
  "type": "tts_chunk",
  "audio": "base64_encoded_audio",
  "chunk_index": 0
}
```

Fin TTS:
```json
{
  "type": "tts_complete",
  "total_chunks": 5
}
```

Pong:
```json
{
  "type": "pong"
}
```

Erreur:
```json
{
  "type": "error",
  "message": "error description"
}
```

## Documentation Interactive

Une fois le service démarré, accédez à la documentation Swagger:

```
http://localhost:8002/docs
```

Documentation ReDoc:

```
http://localhost:8002/redoc
```

## Providers TTS

### Kokoro TTS
TTS rapide et de haute qualité avec plusieurs voix pré-entraînées.

### Chatterbox TTS
TTS expressif avec contrôle fin de l'intonation et des émotions.
Supporte les audio prompts pour guider le style vocal.

### F5-TTS
Clonage vocal zero-shot. Peut reproduire n'importe quelle voix avec un échantillon audio de référence.

### Mémoire GPU insuffisante

Réduisez la taille des modèles ou activez la quantization dans `.env`:
```env
STT_USE_QUANTIZATION=true
```