# D:\Jinvexa\Agents\AssignmentGeneratorAgent.py

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import random
from Agents.BaseAgent import BaseAgent


class AssignmentGeneratorAgent(BaseAgent):
    """
    Generates personalized assignments from lesson content.
    Automatically configures based on course complexity and user level.
    """

    def __init__(
        self,
        llm_client: Any,
        memory_handler: Any,
        config: Optional[Dict] = None
    ):
        super().__init__("AssignmentGeneratorAgent", llm_client)
        
        self.llm_client = llm_client
        self.memory = memory_handler
        self.config = config or {}
        
        # Storage directories
        self.assignments_dir = Path("learn_files/assignments")
        self.sessions_dir = self.assignments_dir / "sessions"
        self.results_dir = self.assignments_dir / "results"
        self.templates_dir = self.assignments_dir / "templates"
        
        # Create directories
        self.assignments_dir.mkdir(exist_ok=True)
        self.sessions_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and generate an assignment."""
        session_id = input_data.get("session_id", "")
        user_id = input_data.get("user_id", "user_1")
        return await self.generate_assignment(session_id, user_id)

    def get_session_lessons(self, session_id: str) -> List[Dict]:
        """Get all lessons for a session from the manifest."""
        manifest_file = Path(f"learn_files/manifests/{session_id}_manifest.json")
        
        if not manifest_file.exists():
            return []
        
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            return manifest.get("watch_order", [])
        except:
            return []

    def get_lesson_content(self, lesson_file: str) -> str:
        """Get the full content of a lesson file."""
        try:
            filepath = Path(lesson_file)
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
        except:
            pass
        return ""

    async def generate_assignment(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete assignment with auto-configuration.
        """
        # Get lessons from manifest
        lessons = self.get_session_lessons(session_id)
        
        if not lessons:
            return {"error": f"No lessons found for session {session_id}"}
        
        # Get lesson content
        lesson_contents = []
        total_content_length = 0
        
        for lesson in lessons:
            text_file = lesson.get("text_file", "")
            if text_file:
                content = self.get_lesson_content(text_file)
                if content:
                    lesson_contents.append({
                        "topic": lesson.get("topic", ""),
                        "phase": lesson.get("phase", ""),
                        "content": content,
                        "content_length": len(content)
                    })
                    total_content_length += len(content)
        
        if not lesson_contents:
            return {"error": "Could not extract lesson content"}
        
        # Get user profile for personalization
        user_profile = None
        if self.memory and user_id:
            user_profile = self.memory.load_profile(user_id)
        
        # Auto-configure assignment using LLM
        config = await self._auto_configure_assignment(
            lesson_contents=lesson_contents,
            total_lessons=len(lesson_contents),
            total_content_length=total_content_length,
            user_profile=user_profile
        )
        
        # Generate questions using LLM
        mcq_questions = await self._generate_mcq_questions(
            lesson_contents, 
            config.get("num_mcq", 5),
            config.get("difficulty", "intermediate")
        )
        
        written_questions = await self._generate_written_questions(
            lesson_contents,
            config.get("num_written", 2),
            config.get("difficulty", "intermediate")
        )
        
        # Build assignment
        assignment_id = f"assign_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        assignment = {
            "assignment_id": assignment_id,
            "session_id": session_id,
            "user_id": user_id,
            "generated_at": datetime.now().isoformat(),
            "configuration": config,
            "total_questions": config.get("num_mcq", 5) + config.get("num_written", 2),
            "questions": {
                "mcq": mcq_questions,
                "written": written_questions
            },
            "time_limit_minutes": self._calculate_time_limit(
                config.get("num_mcq", 5),
                config.get("num_written", 2)
            ),
            "passing_score": config.get("passing_score", 70)
        }
        
        # Save assignment
        self._save_assignment(assignment)
        
        return assignment

    async def _auto_configure_assignment(
        self,
        lesson_contents: List[Dict],
        total_lessons: int,
        total_content_length: int,
        user_profile: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to automatically configure assignment based on course.
        """
        # Get user level if available
        user_level = "beginner"
        user_goals = ""
        previous_performance = ""
        
        if user_profile:
            user_level = user_profile.preferred_depth or "beginner"
            if user_profile.goals:
                user_goals = f"User goals: {', '.join(user_profile.goals)}"
            
            # Check previous assignment performance
            if hasattr(user_profile, 'assignments') and user_profile.assignments:
                last_scores = [a.get('score', 0) for a in user_profile.assignments[-3:]]
                if last_scores:
                    avg_last = sum(last_scores) / len(last_scores)
                    previous_performance = f"Previous average score: {avg_last:.1f}%"
        
        # Prepare course summary
        topics = [l.get("topic", "") for l in lesson_contents if l.get("topic")]
        phases = list(set([l.get("phase", "") for l in lesson_contents if l.get("phase")]))
        
        prompt = f"""
You are an expert educational assessment designer. Configure an assignment for this course.

Course Summary:
- Total Lessons: {total_lessons}
- Topics: {', '.join(topics[:10])}
- Phases: {', '.join(phases)}
- Total Content Length: {total_content_length} characters
- User Level: {user_level}
{user_goals}
{previous_performance}

Based on this information, determine the optimal assignment configuration:

1. Number of MCQ questions (3-10): More for complex courses, fewer for simple ones
2. Number of written questions (1-4): More for advanced/deep topics
3. Difficulty level: beginner, intermediate, or advanced
4. Passing score (60-80%): Based on difficulty and user level

Consider:
- Course complexity (more phases → more questions)
- User level (beginner → more MCQs, advanced → more written)
- Content depth (deep content → more written questions)
- Previous performance (if available)

Return ONLY a JSON object in this exact format:
{{
    "num_mcq": 5,
    "num_written": 2,
    "difficulty": "intermediate",
    "passing_score": 70,
    "reasoning": "Brief explanation of why this configuration was chosen"
}}

JSON:
"""
        
        try:
            response = await self.llm_client.complete_with_json(prompt)
            if response and isinstance(response, dict):
                # Validate and sanitize
                return {
                    "num_mcq": max(3, min(10, response.get("num_mcq", 5))),
                    "num_written": max(1, min(4, response.get("num_written", 2))),
                    "difficulty": response.get("difficulty", "intermediate") if response.get("difficulty") in ["beginner", "intermediate", "advanced"] else "intermediate",
                    "passing_score": max(60, min(80, response.get("passing_score", 70))),
                    "reasoning": response.get("reasoning", "Auto-configured based on course content.")
                }
        except Exception as e:
            print(f"⚠️ Auto-configuration error: {e}")
        
        # Fallback: intelligent default based on content
        return self._fallback_configuration(total_lessons, total_content_length, user_level)

    def _fallback_configuration(
        self,
        total_lessons: int,
        total_content_length: int,
        user_level: str
    ) -> Dict[str, Any]:
        """
        Intelligent fallback configuration based on course metrics.
        """
        # More lessons → more questions
        if total_lessons >= 10:
            num_mcq = 8
            num_written = 3
        elif total_lessons >= 5:
            num_mcq = 6
            num_written = 2
        else:
            num_mcq = 4
            num_written = 1
        
        # More content → more questions
        if total_content_length > 50000:
            num_mcq = min(10, num_mcq + 2)
            num_written = min(4, num_written + 1)
        
        # Difficulty based on user level
        difficulty_map = {
            "beginner": "beginner",
            "intermediate": "intermediate",
            "advanced": "advanced"
        }
        difficulty = difficulty_map.get(user_level, "intermediate")
        
        # Passing score based on difficulty
        passing_score_map = {
            "beginner": 65,
            "intermediate": 70,
            "advanced": 75
        }
        passing_score = passing_score_map.get(difficulty, 70)
        
        return {
            "num_mcq": num_mcq,
            "num_written": num_written,
            "difficulty": difficulty,
            "passing_score": passing_score,
            "reasoning": f"Auto-configured: {total_lessons} lessons, {total_content_length} chars, user level: {user_level}"
        }

    async def _generate_mcq_questions(
        self,
        lesson_contents: List[Dict],
        num_questions: int,
        difficulty: str
    ) -> List[Dict]:
        """
        Generate MCQ questions using LLM.
        """
        # Prepare lesson summaries
        lesson_summaries = []
        for lesson in lesson_contents[:5]:  # Limit to 5 lessons to avoid token overflow
            topic = lesson.get("topic", "")
            content = lesson.get("content", "")[:2000]  # Limit content
            lesson_summaries.append(f"Topic: {topic}\nContent: {content[:500]}...")
        
        combined_content = "\n\n".join(lesson_summaries)
        
        prompt = f"""
You are a expert question generator. Based on the following course content, generate {num_questions} multiple-choice questions.

Course Content:
{combined_content}

Difficulty Level: {difficulty}

For each question, provide:
1. A clear question
2. 4 options (A, B, C, D)
3. The correct answer (A, B, C, or D)
4. The topic it relates to
5. Brief explanation of why the answer is correct

Return ONLY a JSON array in this exact format:
[
    {{
        "id": "mcq_1",
        "question": "Your question here?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": 0,
        "topic": "Related Topic",
        "explanation": "Why this is correct"
    }}
]

The correct_answer should be the index (0-3) of the correct option.

JSON:
"""
        
        try:
            response = await self.llm_client.complete(prompt)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                if isinstance(questions, list):
                    return questions[:num_questions]
        except Exception as e:
            print(f"❌ MCQ generation error: {e}")
        
        # Fallback questions if LLM fails
        return self._fallback_mcq_questions(lesson_contents, num_questions)

    async def _generate_written_questions(
        self,
        lesson_contents: List[Dict],
        num_questions: int,
        difficulty: str
    ) -> List[Dict]:
        """
        Generate written/essay questions using LLM.
        """
        lesson_summaries = []
        for lesson in lesson_contents[:5]:
            topic = lesson.get("topic", "")
            content = lesson.get("content", "")[:2000]
            lesson_summaries.append(f"Topic: {topic}\nContent: {content[:500]}...")
        
        combined_content = "\n\n".join(lesson_summaries)
        
        prompt = f"""
You are a expert question generator. Based on the following course content, generate {num_questions} written/essay questions.

Course Content:
{combined_content}

Difficulty Level: {difficulty}

For each question, provide:
1. A clear, open-ended question
2. The topic it relates to
3. Maximum score (out of 10)
4. Grading rubric with criteria

Return ONLY a JSON array in this exact format:
[
    {{
        "id": "written_1",
        "question": "Your open-ended question here?",
        "topic": "Related Topic",
        "max_score": 10,
        "rubric": {{
            "clarity": 3,
            "correctness": 4,
            "examples": 3
        }}
    }}
]

JSON:
"""
        
        try:
            response = await self.llm_client.complete(prompt)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                if isinstance(questions, list):
                    return questions[:num_questions]
        except Exception as e:
            print(f"❌ Written question generation error: {e}")
        
        return self._fallback_written_questions(lesson_contents, num_questions)

    def _fallback_mcq_questions(self, lesson_contents: List[Dict], num_questions: int) -> List[Dict]:
        """Fallback MCQ questions."""
        questions = []
        topics = [l.get("topic", "Unknown") for l in lesson_contents if l.get("topic")]
        
        for i in range(min(num_questions, len(topics))):
            topic = topics[i % len(topics)]
            questions.append({
                "id": f"mcq_{i+1}",
                "question": f"What is a key concept in {topic}?",
                "options": ["Option A - Correct", "Option B", "Option C", "Option D"],
                "correct_answer": 0,
                "topic": topic,
                "explanation": "This is the fundamental concept of the topic."
            })
        
        return questions

    def _fallback_written_questions(self, lesson_contents: List[Dict], num_questions: int) -> List[Dict]:
        """Fallback written questions."""
        questions = []
        topics = [l.get("topic", "Unknown") for l in lesson_contents if l.get("topic")]
        
        for i in range(min(num_questions, len(topics))):
            topic = topics[i % len(topics)]
            questions.append({
                "id": f"written_{i+1}",
                "question": f"Explain the importance of {topic} in detail. Provide examples.",
                "topic": topic,
                "max_score": 10,
                "rubric": {
                    "clarity": 3,
                    "correctness": 4,
                    "examples": 3
                }
            })
        
        return questions

    def _calculate_time_limit(self, num_mcq: int, num_written: int) -> int:
        """Calculate time limit based on question count."""
        # 1 minute per MCQ, 5 minutes per written
        return (num_mcq * 1) + (num_written * 5) + 5  # +5 minutes buffer

    def _save_assignment(self, assignment: Dict):
        """Save assignment to file."""
        session_id = assignment.get("session_id", "unknown")
        assignment_id = assignment.get("assignment_id", "unknown")
        
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        assignment_file = session_dir / f"{assignment_id}.json"
        with open(assignment_file, 'w', encoding='utf-8') as f:
            json.dump(assignment, f, indent=2, ensure_ascii=False)
        
        return str(assignment_file)

    def get_assignment(self, assignment_id: str) -> Optional[Dict]:
        """Get assignment by ID."""
        # Search in all session directories
        if self.sessions_dir.exists():
            for session_dir in self.sessions_dir.iterdir():
                if session_dir.is_dir():
                    assignment_file = session_dir / f"{assignment_id}.json"
                    if assignment_file.exists():
                        with open(assignment_file, 'r', encoding='utf-8') as f:
                            return json.load(f)
        return None

    def list_assignments(self, session_id: str) -> List[Dict]:
        """List all assignments for a session."""
        assignments = []
        session_dir = self.sessions_dir / session_id
        
        if session_dir.exists():
            for file_path in session_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        assignments.append({
                            "assignment_id": data.get("assignment_id", ""),
                            "generated_at": data.get("generated_at", ""),
                            "total_questions": data.get("total_questions", 0),
                            "difficulty": data.get("difficulty", "intermediate")
                        })
                except:
                    pass
        
        return sorted(assignments, key=lambda x: x.get("generated_at", ""), reverse=True)