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
        """Mode 1: Goal-based learning with memory"""
        print("\n" + "="*60)
        print("📚 MODE 1: Goal-Based Learning")
        print("="*60)
        
        goal = input("\n🎯 What do you want to learn? (e.g., 'I want to become an AI Engineer'): ")
        if not goal or goal.strip() == "":
            print("❌ Please enter a learning goal.")
            return
        
        user_id = input("\n👤 Enter your user ID (default: user_1): ").strip() or "user_1"
        
        # Load existing profile from memory
        profile = self.memory.load_profile(user_id)
        if profile:
            print(f"✅ Loaded existing profile for '{user_id}' ({len(profile.known_concepts)} concepts)")
        else:
            profile = UserProfile(user_id=user_id)
            print(f"🆕 Created new profile for '{user_id}'")
        
        print("\n🔄 Analyzing your goal and creating personalized learning plan...\n")
        
        try:
            conversation, plan, session_id = await self.learning_discovery.process_goal_mode_with_memory(
                user_id=user_id,
                goal_statement=goal,
                user_profile=profile
            )
            
            # Display conversation
            self._display_conversation(conversation)
            
            # Display plan
            self._display_plan(plan)
            
            # Show session info
            print(f"\n💾 Session ID: {session_id[:20]}...")
            print(f"💾 Profile: {len(profile.known_concepts)} concepts saved")
            
            # Show memory stats
            stats = self.memory.get_user_stats(user_id)
            print(f"\n📊 Learning Stats:")
            print(f"   Sessions: {stats['total_sessions']} | Messages: {stats['total_conversations']}")
            if stats.get('knowledge_progress'):
                print(f"   Known: {stats['knowledge_progress'].get('known_concepts', 0)} concepts")
            
            # Ask about continuing conversation
            continue_choice = input("\n🔄 Continue conversation? (y/n): ").lower()
            if continue_choice in ['y', 'yes']:
                await self._continue_conversation(session_id)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    async def run_mode_2_reference(self):
        """Mode 2: Reference-based learning with memory"""
        print("\n" + "="*60)
        print("📚 MODE 2: Reference-Based Learning")
        print("="*60)
        
        source = input("\n📎 Enter URL, file path, or YouTube link: ").strip()
        if not source:
            print("❌ Please provide a valid source.")
            return
        
        # Test extraction
        print("\n🔄 Extracting content...")
        try:
            result = self.data_extractor.extract(source)
            source_type = result.get('type', 'unknown')
            print(f"✅ Type: {source_type}")
            
            # Check if there was an error
            if result.get('error'):
                print(f"⚠️  Warning: {result['error']}")
            
            # Show preview of extracted content
            data = result.get('data', {})
            if isinstance(data, dict):
                content = data.get('content', '')
                if isinstance(content, str) and len(content) > 0:
                    preview = content[:300] + "..." if len(content) > 300 else content
                    print(f"\n📄 Content preview:\n{preview}\n")
            elif isinstance(data, str):
                preview = data[:300] + "..." if len(data) > 300 else data
                print(f"\n📄 Content preview:\n{preview}\n")
                
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return
        
        user_id = input("\n👤 Enter user ID (default: user_1): ").strip() or "user_1"
        
        # Load existing profile from memory
        profile = self.memory.load_profile(user_id)
        if profile:
            print(f"✅ Loaded existing profile for '{user_id}' ({len(profile.known_concepts)} concepts)")
        else:
            profile = UserProfile(user_id=user_id)
            print(f"🆕 Created new profile for '{user_id}'")
        
        print("\n🔄 Analyzing your reference and creating personalized learning plan...\n")
        
        try:
            conversation, plan, session_id = await self.learning_discovery.process_reference_mode_with_memory(
                user_id=user_id,
                source=source,
                user_profile=profile
            )
            
            # Display conversation
            self._display_conversation(conversation)
            
            # Display plan
            self._display_plan(plan)
            
            print(f"\n💾 Session ID: {session_id[:20]}...")
            print(f"💾 Profile: {len(profile.known_concepts)} concepts saved")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def _continue_conversation(self, session_id: str):
        """Continue an existing conversation"""
        print("\n" + "="*60)
        print("💬 CONTINUING CONVERSATION")
        print("="*60)
        print("Type 'quit' to end.\n")
        
        while True:
            user_msg = input("👤 You: ").strip()
            if user_msg.lower() in ['quit', 'exit', 'done']:
                break
            
            if not user_msg:
                continue
            
            result = self.learning_discovery.continue_conversation_with_memory(
                session_id, user_msg
            )
            
            if "error" in result:
                print(f"❌ {result['error']}")
                break
            
            history = result.get("conversation_history", [])
            if history:
                last_msg = history[-1]
                print(f"🤖 Agent: {last_msg['message']}")
    
    def _display_conversation(self, conversation):
        """Display the conversation"""
        print("\n" + "="*60)
        print("💬 DISCOVERY CONVERSATION")
        print("="*60)
        for msg in conversation:
            role = "🤖 Agent" if msg["role"] == "agent" else "👤 You"
            print(f"\n[{role}]: {msg['message']}")
    
    def _display_plan(self, plan):
        """Display the learning plan"""
        print("\n" + "="*60)
        print("📚 YOUR PERSONALIZED LEARNING PLAN")
        print("="*60)
        print(f"\n📌 Topic: {plan.main_topic}")
        print(f"🎯 Goal: {plan.goal}")
        print(f"⏱️ Time: {plan.estimated_time_hours} hours")
        
        if plan.knowledge_gaps:
            print("\n📖 Knowledge Gaps:")
            for gap in plan.knowledge_gaps[:5]:
                print(f"  • {gap}")
        
        if plan.roadmap:
            print("\n🗺️ Roadmap:")
            for phase in plan.roadmap:
                print(f"\n  📍 Phase {phase.get('phase_number')}: {phase.get('title')}")
                print(f"     {phase.get('description', '')}")
                if phase.get('topics'):
                    print(f"     Topics: {', '.join(phase.get('topics', []))}")
                print(f"     ⏱️ {phase.get('estimated_hours', 0)} hours")
        
        if plan.projects:
            print("\n🛠️ Projects:")
            for project in plan.projects[:2]:
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
            print("  1. Goal-Based Learning")
            print("  2. Reference-Based Learning")
            print("  3. View Stats")
            print("  4. View Sessions")
            print("  5. Continue Conversation")
            print("  6. Model Info")
            print("  7. Exit")
            
            try:
                choice = input("\nEnter choice (1-7): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break
            
            if choice == "1":
                await self.run_mode_1_goal()
            elif choice == "2":
                await self.run_mode_2_reference()
            elif choice == "3":
                await self.run_show_stats()
            elif choice == "4":
                await self.run_view_sessions()
            elif choice == "5":
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
            elif choice == "6":
                await self.run_model_info()
            elif choice == "7":
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