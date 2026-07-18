# D:\Jinvexa\app.py

import sys
import os
import asyncio
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import modules
from DataHandle.Utils.DataExtractor import DataExtract
from Agents.ConceptExtractionAgent import ConceptExtractionAgent
from Agents.DependencyAgent import DependencyAgent
from Agents.KnowledgeGapAgent import KnowledgeGapAgent
from Agents.LearningDiscoveryAgent import LearningDiscoveryAgent
from Agents.MemoryHandler import MemoryHandler
from Agents.TeachingAgent import TeachingAgent
from Agents.AssignmentGeneratorAgent import AssignmentGeneratorAgent
from Agents.AssignmentEvaluatorAgent import AssignmentEvaluatorAgent
from Agents.AssignmentTrackerAgent import AssignmentTrackerAgent
from Models.UserProfile import UserProfile


# ==================== CONFIGURATION ====================

# Load environment variables
load_dotenv()

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")

# Storage Configuration
STORAGE_DIR = os.getenv("STORAGE_DIR", "memory_storage")
STORAGE_TYPE = os.getenv("STORAGE_TYPE", "json")

# Suppress httpx logs
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# ==================== OLLAMA LLM CLIENT ====================

class OllamaLLMClient:
    """
    Ollama LLM Client - The brain for all agents.
    """

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        
        # Import ollama
        import ollama
        self.ollama = ollama
        
        print(f"🧠 Using Ollama model: {self.model}")
    
    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a completion request to Ollama.
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            import asyncio
            response = await asyncio.to_thread(
                self.ollama.chat,
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.3,
                    "num_predict": 2000
                }
            )
            
            # Handle response
            if hasattr(response, 'message'):
                if hasattr(response.message, 'content'):
                    return response.message.content
                elif isinstance(response.message, dict):
                    return response.message.get('content', '')
            elif isinstance(response, dict):
                return response.get('message', {}).get('content', '')
            
            return str(response)
            
        except Exception as e:
            print(f"❌ Ollama API Error: {e}")
            return json.dumps({
                "main_topic": "General Learning",
                "subtopics": [],
                "domain": "General",
                "difficulty": "intermediate",
                "keywords": []
            })
    
    async def complete_with_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a completion request and parse response as JSON.
        """
        response = await self.complete(prompt, system_prompt)
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        try:
            return json.loads(response)
        except:
            return {}


# ==================== JINVEXA APP ====================

class JinvexaApp:
    """Main application class for Jinvexa Learning Platform"""
    
    def __init__(self):
        # Create necessary directories
        Path("profiles").mkdir(exist_ok=True)
        Path(STORAGE_DIR).mkdir(exist_ok=True)
        
        # Initialize Ollama LLM Client
        self.llm_client = OllamaLLMClient()
        
        # Initialize Memory Handler
        self.memory = MemoryHandler(
            storage_dir=STORAGE_DIR,
            storage_type=STORAGE_TYPE
        )
        
        # Initialize DataExtractor with the llm_client
        self.data_extractor = DataExtract(llm_client=self.llm_client)
        
        # Initialize Agents with Ollama LLM
        self.concept_extractor = ConceptExtractionAgent(self.llm_client)
        self.dependency_agent = DependencyAgent(self.llm_client)
        self.knowledge_gap_agent = KnowledgeGapAgent(self.llm_client)
        
        # Initialize Learning Discovery Agent with Memory
        self.learning_discovery = LearningDiscoveryAgent(
            data_extractor=self.data_extractor,
            concept_extractor=self.concept_extractor,
            dependency_agent=self.dependency_agent,
            knowledge_gap_agent=self.knowledge_gap_agent,
            llm_client=self.llm_client,
            memory_handler=self.memory
        )
        
        # Initialize Teaching Agent
        self.teaching_agent = TeachingAgent(
            llm_client=self.llm_client,
            memory_handler=self.memory
        )
        
        # Initialize Assignment Agents
        self.assignment_generator = AssignmentGeneratorAgent(
            llm_client=self.llm_client,
            memory_handler=self.memory
        )
        
        self.assignment_evaluator = AssignmentEvaluatorAgent(
            llm_client=self.llm_client,
            memory_handler=self.memory,
            assignment_generator=self.assignment_generator
        )
        
        self.assignment_tracker = AssignmentTrackerAgent(
            llm_client=self.llm_client,
            memory_handler=self.memory
        )
        
        self.current_profile = None
    
    def display_banner(self):
        """Display application banner"""
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║         🧠  J I N V E X A  L E A R N I N G  A I               ║
║                                                                  ║
║     "Tell me what you want to become.                           ║
║      I'll build your university."                               ║
║                                                                  ║
║     🤖 Brain: {self.llm_client.model:<20}                       ║
║     💾 Storage: {STORAGE_TYPE.upper():<8}                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """)
    
    async def run_mode_1_goal(self):
        """Mode 1: Goal-based learning with interactive conversation"""
        print("\n" + "="*60)
        print("📚 MODE 1: Goal-Based Learning")
        print("="*60)
        
        goal = input("\n🎯 What do you want to learn? (e.g., 'I want to become an AI Engineer'): ")
        if not goal or goal.strip() == "":
            print("❌ Please enter a learning goal.")
            return
        
        user_id = input("\n👤 Enter your user ID (default: user_1): ").strip() or "user_1"
        
        # Load existing profile
        profile = self.memory.load_profile(user_id)
        if profile:
            print(f"✅ Loaded existing profile for '{user_id}' ({len(profile.known_concepts)} concepts)")
        else:
            profile = UserProfile(user_id=user_id)
            print(f"🆕 Created new profile for '{user_id}'")
        
        print("\n🔄 Analyzing your goal and creating personalized learning plan...\n")
        
        try:
            # Get conversation start with questions
            conversation, questions, session_id = await self.learning_discovery.start_goal_conversation(
                user_id=user_id,
                goal_statement=goal,
                user_profile=profile
            )
            
            # Display initial conversation - handle safely
            print("\n" + "="*60)
            print("💬 DISCOVERY CONVERSATION")
            print("="*60)
            
            for msg in conversation:
                role = "🤖 Agent" if msg.get("role") == "agent" else "👤 You"
                message = msg.get("message", msg.get("question", str(msg)))
                print(f"\n[{role}]: {message}")
            
            # Interactive loop
            print("\n" + "="*60)
            print("💬 INTERACTIVE DISCOVERY")
            print("="*60)
            print("Type 'quit' to skip questions and generate plan\n")
            
            answered = 0
            total_questions = len(questions) if questions else 0
            
            while answered < total_questions:
                user_input = input("\n👤 Your answer: ").strip()
                if user_input.lower() in ['quit', 'exit', 'done']:
                    break
                
                if not user_input:
                    continue
                
                # Process answer
                result = await self.learning_discovery.process_answer(
                    session_id=session_id,
                    answer=user_input,
                    question_index=answered
                )
                
                if result.get("type") == "plan_ready":
                    print("\n✅ Great! Learning plan generated!\n")
                    plan_data = result.get("plan")
                    if plan_data:
                        self._display_plan(plan_data)
                    else:
                        print("⚠️ Plan generated but data is empty.")
                    break
                elif result.get("type") == "next_question":
                    answered += 1
                    print(f"\n🤖 Agent: {result.get('question')}")
                    if result.get('options'):
                        print(f"   Options: {', '.join(result['options'])}")
                    if result.get('progress'):
                        print(f"   Progress: {result['progress']}")
                elif result.get("error"):
                    print(f"❌ {result.get('error')}")
                    break
            
            # Show memory stats
            stats = self.memory.get_user_stats(user_id)
            print(f"\n📊 Learning Stats:")
            print(f"   Sessions: {stats['total_sessions']} | Messages: {stats['total_conversations']}")
            if stats.get('knowledge_progress'):
                print(f"   Known Concepts: {stats['knowledge_progress'].get('known_concepts', 0)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def run_mode_2_reference(self):
        """Mode 2: Reference-Based Learning - User provides a document/video/URL"""
        print("\n" + "="*60)
        print("📚 MODE 2: Reference-Based Learning")
        print("="*60)
        
        source = input("\n📎 Enter URL, file path, or YouTube link: ").strip()
        if not source:
            print("❌ Please provide a valid source.")
            return
        
        # Extract content
        print("\n🔄 Extracting content...")
        try:
            result = self.data_extractor.extract(source)
            source_type = result.get('type', 'unknown')
            print(f"✅ Type: {source_type}")
            
            # Show content preview
            data = result.get('data', {})
            if isinstance(data, dict):
                content = data.get('content', '')
                if content and len(content) > 100:
                    preview = content[:500] + "..." if len(content) > 500 else content
                    print(f"\n📄 Content preview ({len(content)} chars):\n{preview[:300]}...\n")
                elif content:
                    print(f"\n📄 Content length: {len(content)} chars")
                else:
                    print("⚠️  No content extracted")
                    
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return
        
        # Get user ID
        user_id = input("\n👤 Enter your user ID (default: user_1): ").strip() or "user_1"
        
        # Load or create profile
        profile = self.memory.load_profile(user_id)
        if profile:
            print(f"✅ Loaded existing profile for '{user_id}' ({len(profile.known_concepts)} concepts)")
        else:
            profile = UserProfile(user_id=user_id)
            print(f"🆕 Created new profile for '{user_id}'")
        
        print("\n🔄 Analyzing your reference and creating personalized learning plan...\n")
        
        try:
            # Start interactive reference conversation
            conversation, questions, session_id = await self.learning_discovery.start_reference_conversation(
                user_id=user_id,
                source=source,
                user_profile=profile
            )
            
            # Display initial conversation
            print("\n" + "="*60)
            print("💬 DISCOVERY CONVERSATION")
            print("="*60)
            
            for msg in conversation:
                role = "🤖 Agent" if msg.get("role") == "agent" else "👤 You"
                message = msg.get("message", msg.get("question", str(msg)))
                print(f"\n[{role}]: {message}")
            
            # Interactive loop
            print("\n" + "="*60)
            print("💬 INTERACTIVE DISCOVERY")
            print("="*60)
            print("Type 'quit' to skip questions and generate plan\n")
            
            answered = 0
            total_questions = len(questions) if questions else 0
            
            while answered < total_questions:
                user_input = input("\n👤 Your answer: ").strip()
                if user_input.lower() in ['quit', 'exit', 'done']:
                    break
                
                if not user_input:
                    continue
                
                # Process answer
                result = await self.learning_discovery.process_answer(
                    session_id=session_id,
                    answer=user_input,
                    question_index=answered
                )
                
                if result.get("type") == "plan_ready":
                    print("\n✅ Great! Learning plan generated!\n")
                    plan_data = result.get("plan")
                    if plan_data:
                        self._display_plan(plan_data)
                    else:
                        print("⚠️ Plan generated but data is empty.")
                    break
                elif result.get("type") == "next_question":
                    answered += 1
                    print(f"\n🤖 Agent: {result.get('question')}")
                    if result.get('options'):
                        print(f"   Options: {', '.join(result['options'])}")
                    if result.get('progress'):
                        print(f"   Progress: {result['progress']}")
                elif result.get("error"):
                    print(f"❌ {result.get('error')}")
                    break
            
            # Show memory stats
            stats = self.memory.get_user_stats(user_id)
            print(f"\n📊 Learning Stats:")
            print(f"   Sessions: {stats['total_sessions']} | Messages: {stats['total_conversations']}")
            if stats.get('knowledge_progress'):
                print(f"   Known Concepts: {stats['knowledge_progress'].get('known_concepts', 0)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _display_conversation(self, conversation):
        """Display the conversation with proper error handling"""
        print("\n" + "="*60)
        print("💬 DISCOVERY CONVERSATION")
        print("="*60)
        
        if not conversation:
            print("No conversation history available.")
            return
        
        for msg in conversation:
            role = "🤖 Agent" if msg.get("role") == "agent" else "👤 You"
            
            # Handle different message formats
            if "question" in msg:
                # It's a question object
                message = msg.get("question", "")
            elif "message" in msg:
                message = msg.get("message", "")
            elif "content" in msg:
                message = msg.get("content", "")
            else:
                # Try to get any string value
                message = str(msg) if not isinstance(msg, dict) else "Unknown message"
            
            print(f"\n[{role}]: {message}")
    
    def _display_plan(self, plan):
        """Display the learning plan with safety checks"""
        if not plan:
            print("\n❌ No learning plan generated. Please try again.")
            return
        
        print("\n" + "="*60)
        print("📚 YOUR PERSONALIZED LEARNING PLAN")
        print("="*60)
        
        # Safely access attributes - handle both dict and object
        if isinstance(plan, dict):
            main_topic = plan.get('main_topic', 'Unknown Topic')
            goal = plan.get('goal', 'Master the topic')
            estimated_time = plan.get('estimated_time_hours', 0)
            knowledge_gaps = plan.get('knowledge_gaps', [])
            roadmap = plan.get('roadmap', [])
            projects = plan.get('projects', [])
        else:
            # It's a LearningPlan object
            main_topic = getattr(plan, 'main_topic', 'Unknown Topic')
            goal = getattr(plan, 'goal', 'Master the topic')
            estimated_time = getattr(plan, 'estimated_time_hours', 0)
            knowledge_gaps = getattr(plan, 'knowledge_gaps', [])
            roadmap = getattr(plan, 'roadmap', [])
            projects = getattr(plan, 'projects', [])
        
        print(f"\n📌 Topic: {main_topic}")
        print(f"🎯 Goal: {goal}")
        print(f"⏱️ Time: {estimated_time} hours")
        
        if knowledge_gaps:
            print("\n📖 Knowledge Gaps:")
            for gap in knowledge_gaps[:5]:
                print(f"  • {gap}")
        
        if roadmap:
            print("\n🗺️ Roadmap:")
            for phase in roadmap:
                print(f"\n  📍 Phase {phase.get('phase_number')}: {phase.get('title')}")
                print(f"     {phase.get('description', '')}")
                if phase.get('topics'):
                    print(f"     Topics: {', '.join(phase.get('topics', []))}")
                print(f"     ⏱️ {phase.get('estimated_hours', 0)} hours")
        
        if projects:
            print("\n🛠️ Projects:")
            for project in projects[:2]:
                print(f"  • {project.get('title', '')}")
        
        print("\n" + "="*60)
        print("✅ Learning plan generated successfully!")
        print("="*60)
    
    async def run_show_stats(self):
        """Show user statistics"""
        print("\n" + "="*60)
        print("📊 USER LEARNING STATISTICS")
        print("="*60)
        
        user_id = input("\n👤 Enter user ID (default: user_1): ").strip() or "user_1"
        
        stats = self.memory.get_user_stats(user_id)
        
        print(f"\n📊 Stats for: {user_id}")
        print(f"   Sessions: {stats['total_sessions']}")
        print(f"   Messages: {stats['total_conversations']}")
        
        if stats.get('knowledge_progress'):
            print(f"   Known Concepts: {stats['knowledge_progress'].get('known_concepts', 0)}")
            topics = stats['knowledge_progress'].get('topics', [])
            if topics:
                print(f"   Topics: {', '.join(topics[:5])}")
        
        if stats.get('learning_plans'):
            print(f"\n📚 Learning Plans:")
            for plan in stats['learning_plans'][:3]:
                print(f"   • {plan['topic']} - {plan['goal']} ({plan['estimated_hours']}h)")
    
    async def run_view_sessions(self):
        """View all sessions for a user"""
        print("\n" + "="*60)
        print("📋 USER SESSIONS")
        print("="*60)
        
        user_id = input("\n👤 Enter user ID (default: user_1): ").strip() or "user_1"
        
        sessions = self.memory.get_user_sessions(user_id)
        
        if not sessions:
            print(f"\nNo sessions found for: {user_id}")
            return
        
        print(f"\n📋 Found {len(sessions)} sessions")
        print("-" * 50)
        
        for i, session in enumerate(sessions, 1):
            print(f"{i}. {session.session_id[:20]}... ({session.mode})")
            print(f"   Created: {session.created_at[:19]}")
            print(f"   Messages: {len(session.conversation_history)}")
            # Handle None values
            if session.concepts and isinstance(session.concepts, dict):
                print(f"   Topic: {session.concepts.get('main_topic', 'Unknown')}")
            else:
                print(f"   Topic: Unknown")
            print()
        
        choice = input("🔍 Enter session number to view details (or Enter to skip): ")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                session = sessions[idx]
                summary = self.learning_discovery.get_session_summary(session.session_id)
                print("\n📋 Session Details:")
                print(json.dumps(summary, indent=2, default=str))
    
    async def _handle_continue_conversation(self):
        """Handle continue conversation flow"""
        user_id = input("\n👤 Enter user ID (default: user_1): ").strip() or "user_1"
        sessions = self.memory.get_user_sessions(user_id)
        if sessions:
            print("\n📋 Recent sessions:")
            for i, s in enumerate(sessions[:5], 1):
                print(f"  {i}. {s.session_id[:20]}... ({s.mode}) - {s.created_at[:19]}")
            session_num = input("\nEnter session number: ")
            if session_num.isdigit():
                idx = int(session_num) - 1
                if 0 <= idx < len(sessions):
                    await self._continue_conversation(sessions[idx].session_id)
        else:
            print("❌ No sessions found.")
    
    async def run_mode_teaching(self):
        """Mode 3: Teaching Layer - Generate lessons from learning plans"""
        print("\n" + "="*60)
        print("📚 MODE 3: Teaching Layer")
        print("="*60)
        
        # Get user ID first
        user_id = input("\n👤 Enter your user ID (default: user_1): ").strip() or "user_1"
        
        # List available sessions
        all_sessions = self.teaching_agent.list_available_sessions()
        
        if not all_sessions:
            print("❌ No sessions with learning plans found.")
            print("   Please create a learning plan first (Mode 1 or 2).")
            return
        
        # Filter sessions for this user
        sessions = [s for s in all_sessions if s.get("user_id", "").lower() == user_id.lower()]
        
        if not sessions:
            print(f"❌ No sessions found for user '{user_id}'.")
            print("   Please create a learning plan first (Mode 1 or 2).")
            return
        
        print(f"\n📋 Available sessions for '{user_id}':")
        print("-" * 50)
        for i, session in enumerate(sessions, 1):
            print(f"{i}. Session: {session['session_id'][:20]}...")
            print(f"   Topic: {session.get('main_topic', 'Unknown')}")
            print(f"   Phases: {session.get('phase_count', 0)}")
            print(f"   Total Time: {session.get('total_hours', 0)} hours")
            print()
        
        # Select session
        choice = input("\n🔍 Select session number to generate course: ").strip()
        if not choice.isdigit():
            print("❌ Invalid selection.")
            return
        
        idx = int(choice) - 1
        if idx < 0 or idx >= len(sessions):
            print("❌ Invalid session number.")
            return
        
        selected_session = sessions[idx]
        session_id = selected_session["session_id"]
        
        print(f"\n📚 Generating course for: {selected_session.get('main_topic', 'Unknown')}")
        print(f"   Phases: {selected_session.get('phase_count', 0)}")
        print(f"   Total Time: {selected_session.get('total_hours', 0)} hours")
        
        print("\n🧠 AI is analyzing each topic to decide the best format...")
        print("   (text, female voice, or male voice based on content)\n")
        
        try:
            result = await self.teaching_agent.generate_course_from_session(
                session_id=session_id
            )
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return
            
            # Display format decisions
            if result.get("format_decisions"):
                print("\n" + "="*60)
                print("📋 FORMAT DECISIONS MADE BY AI")
                print("="*60)
                for decision in result["format_decisions"]:
                    output_format = decision.get("output_format", "text")
                    if output_format == "text":
                        emoji = "📄"
                        label = "Text (Self-paced reading)"
                    elif output_format == "female_voice":
                        emoji = "👩"
                        label = "Female Voice (Warm audio)"
                    elif output_format == "male_voice":
                        emoji = "👨"
                        label = "Male Voice (Professional audio)"
                    else:
                        emoji = "📄"
                        label = "Text"
                    print(f"   {emoji} {decision.get('topic')} → {label}")
                    print(f"      💡 {decision.get('reason', '')[:80]}...")
            
            # Display results
            print("\n" + "="*60)
            print("✅ COURSE GENERATION COMPLETE")
            print("="*60)
            
            print(f"\n📌 Course: {result.get('main_topic', 'Unknown')}")
            print(f"📝 Total Lessons: {result.get('total_lessons', 0)}")
            
            # Show manifest/watch order
            manifest = result.get('manifest', [])
            if manifest:
                print("\n📋 WATCH ORDER (from manifest):")
                print("-" * 40)
                for item in manifest[:10]:
                    order = item.get('order', 0)
                    topic = item.get('topic', '')
                    content_type = item.get('content_type', 'text')
                    gender = item.get('gender', '')
                    
                    if content_type == 'audio':
                        icon = f"🎧 ({gender})"
                    else:
                        icon = "📄"
                    
                    print(f"   {order}. {icon} {topic}")
                
                if len(manifest) > 10:
                    print(f"   ... and {len(manifest) - 10} more lessons")
            
            # Show files
            text_files = result.get('text_files', [])
            audio_files = result.get('audio_files', [])
            
            if text_files:
                print(f"\n📄 Text Lessons ({len(text_files)} files):")
                for file in text_files[:5]:
                    print(f"   • {Path(file).name}")
                if len(text_files) > 5:
                    print(f"   ... and {len(text_files) - 5} more")
            
            if audio_files:
                print(f"\n🎧 Audio Lessons ({len(audio_files)} files):")
                for file in audio_files[:5]:
                    print(f"   • {Path(file).name}")
                if len(audio_files) > 5:
                    print(f"   ... and {len(audio_files) - 5} more")
            
            print(f"\n💾 Files saved in: {self.teaching_agent.learn_files_dir}/")
            print(f"   📄 Lessons: {self.teaching_agent.lessons_dir}/")
            print(f"   🎧 Audio: {self.teaching_agent.audio_dir}/")
            print(f"   📋 Manifest: {self.teaching_agent.manifest_dir}/")
            
            # Show manifest location
            manifest_file = self.teaching_agent.manifest_dir / f"{session_id}_manifest.json"
            if manifest_file.exists():
                print(f"\n📋 Manifest saved at: {manifest_file}")
                print(f"   Contains complete watch order with content types")
            
        except Exception as e:
            print(f"❌ Error generating course: {e}")
            import traceback
            traceback.print_exc()

    async def run_mode_assignment(self):
        """Mode 4: Assignment Layer - Auto-configured by AI"""
        print("\n" + "="*60)
        print("📝 MODE 4: Assignment Layer")
        print("="*60)
        
        # Get user ID
        user_id = input("\n👤 Enter your user ID (default: user_1): ").strip() or "user_1"
        
        # List available sessions with completed courses
        sessions = self.teaching_agent.list_available_sessions()
        
        if not sessions:
            print("❌ No sessions with learning plans found.")
            print("   Please complete a course first (Mode 3).")
            return
        
        # Filter sessions that have courses generated
        available_sessions = []
        for session in sessions:
            session_id = session.get("session_id", "")
            manifest_file = Path(f"learn_files/manifests/{session_id}_manifest.json")
            if manifest_file.exists():
                available_sessions.append(session)
        
        if not available_sessions:
            print("❌ No completed courses found.")
            print("   Please generate a course first (Mode 3).")
            return
        
        print("\n📋 Available completed courses:")
        print("-" * 50)
        for i, session in enumerate(available_sessions, 1):
            print(f"{i}. Topic: {session.get('main_topic', 'Unknown')}")
            print(f"   Session: {session['session_id'][:20]}...")
            print(f"   Phases: {session.get('phase_count', 0)}")
            print(f"   Total Time: {session.get('total_hours', 0)} hours")
            print()
        
        # Select session
        choice = input("\n🔍 Select course for assignment: ").strip()
        if not choice.isdigit():
            print("❌ Invalid selection.")
            return
        
        idx = int(choice) - 1
        if idx < 0 or idx >= len(available_sessions):
            print("❌ Invalid session number.")
            return
        
        selected_session = available_sessions[idx]
        session_id = selected_session["session_id"]
        
        print(f"\n📚 Generating assignment for: {selected_session.get('main_topic', 'Unknown')}")
        print("\n🧠 AI is analyzing the course to configure the assignment...")
        
        try:
            # Generate assignment with auto-configuration
            assignment = await self.assignment_generator.generate_assignment(
                session_id=session_id,
                user_id=user_id
            )
            
            if "error" in assignment:
                print(f"❌ Error: {assignment['error']}")
                return
            
            # Display auto-configuration
            config = assignment.get("configuration", {})
            print("\n" + "="*60)
            print("📋 ASSIGNMENT CONFIGURATION (Auto-decided by AI)")
            print("="*60)
            print(f"📝 MCQ Questions: {config.get('num_mcq', 5)}")
            print(f"📝 Written Questions: {config.get('num_written', 2)}")
            print(f"📊 Difficulty: {config.get('difficulty', 'intermediate').upper()}")
            print(f"🎯 Passing Score: {config.get('passing_score', 70)}%")
            print(f"💡 Reasoning: {config.get('reasoning', 'Auto-configured based on course content.')}")
            
            assignment_id = assignment.get("assignment_id", "")
            mcq_questions = assignment.get("questions", {}).get("mcq", [])
            written_questions = assignment.get("questions", {}).get("written", [])
            
            print(f"\n⏱️ Time Limit: {assignment.get('time_limit_minutes', 0)} minutes")
            print(f"📝 Total Questions: {len(mcq_questions) + len(written_questions)}")
            
            # Display MCQ questions
            print("\n" + "="*60)
            print("📌 MULTIPLE CHOICE QUESTIONS")
            print("="*60)
            
            user_answers = {}
            
            for i, q in enumerate(mcq_questions, 1):
                print(f"\n{i}. {q.get('question', '')}")
                print(f"   Topic: {q.get('topic', '')}")
                for j, option in enumerate(q.get('options', [])):
                    print(f"   {chr(65 + j)}. {option}")
                print()
                
                while True:
                    answer = input(f"Your answer ({chr(65)}-{chr(65 + len(q.get('options', [])) - 1)}): ").strip().upper()
                    if answer and answer in [chr(65 + i) for i in range(len(q.get('options', [])))]:
                        user_answers[q.get('id', f"mcq_{i}")] = ord(answer) - 65
                        break
                    else:
                        print(f"❌ Invalid. Please enter {chr(65)}-{chr(65 + len(q.get('options', [])) - 1)}")
            
            # Display written questions
            print("\n" + "="*60)
            print("📌 WRITTEN/ESSAY QUESTIONS")
            print("="*60)
            
            for i, q in enumerate(written_questions, 1):
                print(f"\n{i}. {q.get('question', '')}")
                print(f"   Topic: {q.get('topic', '')}")
                print(f"   Max Score: {q.get('max_score', 10)}")
                print()
                print("   Enter your answer (type 'skip' to skip this question):")
                answer = input("   Answer: ").strip()
                
                if answer.lower() == 'skip':
                    user_answers[q.get('id', f"written_{i}")] = ""
                else:
                    user_answers[q.get('id', f"written_{i}")] = answer
            
            print("\n🔄 Evaluating your answers...")
            
            # Evaluate assignment
            result = await self.assignment_evaluator.evaluate_assignment(
                assignment_id=assignment_id,
                user_answers=user_answers,
                user_id=user_id
            )
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return
            
            # Display results
            print("\n" + "="*60)
            print("📊 ASSIGNMENT RESULTS")
            print("="*60)
            
            total = result.get("scores", {}).get("total", {})
            mcq = result.get("scores", {}).get("mcq", {})
            written = result.get("scores", {}).get("written", {})
            
            print(f"\n📌 Overall Score: {total.get('percentage', 0)}%")
            print(f"📊 Grade: {total.get('grade', 'N/A')}")
            print(f"✅ MCQ: {mcq.get('correct', 0)}/{mcq.get('total', 0)} ({mcq.get('percentage', 0)}%)")
            print(f"📝 Written: {written.get('score', 0)}/{written.get('total', 0)} ({written.get('percentage', 0)}%)")
            print(f"🎯 Passing Score: {assignment.get('passing_score', 70)}%")
            print(f"📈 Status: {'✅ PASSED' if total.get('percentage', 0) >= assignment.get('passing_score', 70) else '❌ NEEDS REVIEW'}")
            
            # Show feedback
            feedback = result.get("feedback", {})
            print(f"\n💬 Feedback: {feedback.get('overall', '')}")
            print(f"📚 Recommendation: {feedback.get('recommendation', '')}")
            
            # Show wrong answers
            wrong_answers = result.get("wrong_answers", [])
            if wrong_answers:
                print("\n❌ Incorrect Answers:")
                for wa in wrong_answers[:3]:
                    if wa.get("type") == "mcq":
                        print(f"   • {wa.get('question', '')}")
                        print(f"     Correct: {wa.get('correct_answer', '')}")
                        print(f"     Your answer: {wa.get('user_answer', '')}")
                        print(f"     Explanation: {wa.get('explanation', '')}")
                    else:
                        print(f"   • {wa.get('question', '')}")
                        print(f"     Score: {wa.get('score', 0)}/{wa.get('max_score', 10)}")
                        print(f"     Feedback: {wa.get('feedback', '')}")
            
            # Show improvement areas
            improvement_areas = result.get("improvement_areas", [])
            if improvement_areas:
                print("\n📚 Areas for Improvement:")
                for area in improvement_areas[:3]:
                    print(f"   • {area}")
            
            # Save progress
            self.assignment_tracker.save_progress_summary(user_id)
            print(f"\n💾 Progress saved for user: {user_id}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    async def run_mode_progress(self):
        """Mode 5: View progress"""
        print("\n" + "="*60)
        print("📊 PROGRESS TRACKING")
        print("="*60)
        
        user_id = input("\n👤 Enter your user ID (default: user_1): ").strip() or "user_1"
        
        progress = self.assignment_tracker.get_user_progress(user_id)
        
        print(f"\n📊 Progress for: {user_id}")
        print("-" * 40)
        print(f"📚 Total Assignments: {progress.get('total_assignments', 0)}")
        print(f"📈 Average Score: {progress.get('average_score', 0)}%")
        print(f"🏆 Best Score: {progress.get('best_score', 0)}%")
        print(f"🎯 Latest Score: {progress.get('latest_score', 0)}%")
        print(f"📊 Latest Grade: {progress.get('latest_grade', 'N/A')}")
        print(f"📉 Trend: {progress.get('trend', 'Insufficient data')}")
        print(f"⭐ Performance: {progress.get('performance', 'No data')}")
        
        # Show certificate eligibility
        eligibility = self.assignment_tracker.get_certificate_eligibility(user_id)
        print(f"\n🎓 Certificate Status: {eligibility.get('status', 'N/A')}")
        if eligibility.get('eligible'):
            print("   ✅ You are eligible for a certificate!")
        else:
            print(f"   Requirements: {eligibility.get('requirement', '')}")
        
        # Show history
        history = progress.get("history", [])
        if history:
            print("\n📋 Recent Assignments:")
            for h in history[:5]:
                date = h.get("evaluated_at", "")[:16] if h.get("evaluated_at") else "Unknown"
                score = h.get("scores", {}).get("total", {}).get("percentage", 0)
                grade = h.get("scores", {}).get("total", {}).get("grade", "N/A")
                print(f"   • {date}: {score}% ({grade})")
        
        # Recommendations
        recommendations = progress.get("recommendations", [])
        if recommendations:
            print("\n💡 Recommendations:")
            for rec in recommendations[:3]:
                print(f"   • {rec}")

    async def run_teaching_status(self):
        """View teaching status for a session"""
        print("\n" + "="*60)
        print("📊 TEACHING STATUS")
        print("="*60)
        
        session_id = input("\n📎 Enter session ID: ").strip()
        if not session_id:
            print("❌ Please enter a session ID.")
            return
        
        # Check if course exists
        status = self.teaching_agent.get_course_status(session_id)
        
        if "error" in status:
            print(f"❌ {status['error']}")
            return
        
        print(f"\n📌 Course: {status.get('main_topic', 'Unknown')}")
        print(f"📝 Total Lessons: {status.get('total_lessons', 0)}")
        print(f"📄 Text Files: {len(status.get('text_files', []))}")
        print(f"🎧 Audio Files: {len(status.get('audio_files', []))}")
        
        # Show recent phases
        if status.get('phases'):
            print("\n📚 Generated Phases:")
            for phase in status['phases']:
                print(f"   • Phase {phase.get('phase_number')}: {phase.get('phase_title')}")
                print(f"     Lessons: {len(phase.get('lessons', []))}")
    
    async def run_model_info(self):
        """Show model information"""
        print("\n" + "="*60)
        print("🤖 OLLAMA MODEL INFO")
        print("="*60)
        print(f"\n📌 Model: {self.llm_client.model}")
    
    async def run(self):
        """Main run loop"""
        self.display_banner()
        
        while True:
            print("\n📚 Choose mode:")
            print("  1. Goal-Based Learning (Discovery)")
            print("  2. Reference-Based Learning (Discovery)")
            print("  3. Teaching Layer (Generate Course)")
            print("  4. Assignment Layer (Take Assignment)")
            print("  5. View Progress")
            print("  6. View Stats")
            print("  7. View Sessions")
            print("  8. Continue Conversation")
            print("  9. Teaching Status")
            print(" 10. Model Info")
            print(" 11. Exit")
            
            try:
                choice = input("\nEnter choice (1-11): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break
            
            if choice == "1":
                await self.run_mode_1_goal()
            elif choice == "2":
                await self.run_mode_2_reference()
            elif choice == "3":
                await self.run_mode_teaching()
            elif choice == "4":
                await self.run_mode_assignment()
            elif choice == "5":
                await self.run_mode_progress()
            elif choice == "6":
                await self.run_show_stats()
            elif choice == "7":
                await self.run_view_sessions()
            elif choice == "8":
                await self._handle_continue_conversation()
            elif choice == "9":
                await self.run_teaching_status()
            elif choice == "10":
                await self.run_model_info()
            elif choice == "11":
                print("\n👋 Thank you for using Jinvexa! Good luck with your learning journey!")
                break
            else:
                print("❌ Invalid choice. Please try again.")


# ==================== MAIN ENTRY POINT ====================

async def main():
    """Main entry point"""
    app = JinvexaApp()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())