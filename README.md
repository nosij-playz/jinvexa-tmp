# 🧠 Jinvexa Learning AI

Jinvexa is an AI-powered personalized learning platform that transforms how you acquire new skills. It doesn't just give you a list of links; it builds a custom "university" tailored to your goals, current knowledge, and preferred learning style.

## 🏗️ System Architecture

The platform is built in layers, moving from raw data extraction to personalized mentorship.

### 1. Input Layer (Data Extraction) ✅
**Status: Complete**
Responsible for ingesting information from various sources:
- 📺 **YouTube**: Transcripts and video metadata.
- 🌐 **Websites**: Article and documentation parsing.
- 📄 **Documents**: PDF and DOCX processing.
- 🖼️ **Images**: OCR for text extraction from images.

### 2. Planning Layer (Discovery) ✅
**Status: Complete**
The "Brain" of the system that analyzes content and user profiles:
- **Concept Extraction**: Identifying core topics and subtopics.
- **Dependency Mapping**: Determining the logical order of learning.
- **Knowledge Gap Analysis**: Comparing user's current knowledge with the goal.
- **Learning Plan Generation**: Creating a structured roadmap.
- **Memory Handler**: Persistent storage of user profiles and session history.

### 3. Teaching Layer (Execution) ✅
**Status: Complete**
Generates full courses from learning plans with AI-driven format selection and parallel processing:
- **AI Format Selection**: LLM decides per-topic whether to generate text, female voice audio, or male voice audio based on content type and learner level.
- **Multithreaded Generation**: Uses `ThreadPoolExecutor` (max 5 workers) to generate all lessons in parallel — dramatically faster than sequential generation.
- **Text Lessons**: Comprehensive markdown-formatted lessons with introduction, core concepts, examples, exercises, and next steps.
- **Audio Lessons**: TTS-generated audio files using `edge-tts` with gender-specific voices (female for warm/beginner topics, male for professional/advanced topics).
- **Structured Output**: All content organized under `learn_files/`:
  - `learn_files/lessons/` — Text lesson files (`.txt`)
  - `learn_files/audio/` — Audio lesson files (`.mp3`)
  - `learn_files/manifests/` — JSON manifest + human-readable watch order
- **Manifest System**: Generates a JSON manifest with complete watch order, content types, file paths, and format reasons — enabling playlist-style consumption.
- **Course Metadata**: Saves course metadata to session storage for status tracking and resumption.

### 4. Assessment Layer (Evaluation) ❌
**Status: Not Built Yet**
*Planned features:*
- Exercise Agent & Code Reviewer.
- Project Checker for practical application.
- Certification tracking.

### 5. Mentoring Layer (Guidance) ❌
**Status: Not Built Yet**
*Planned features:*
- Mentor Agent for high-level guidance.
- Revision Agent for spaced repetition.
- Career Agent for professional alignment.
- Daily Planner for consistency.

---

## 🚀 Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nosij-playz/Jinvexa.git
   cd Jinvexa
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your `.env` file:
   ```env
   OLLAMA_MODEL=gemma4:31b-cloud
   STORAGE_DIR=memory_storage
   STORAGE_TYPE=json
   ```

### Running the App
```bash
python app.py
```