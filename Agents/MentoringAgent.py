# D:\Jinvexa\Agents\MentoringAgent.py

import json
import sqlite3
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import re
import asyncio
from Agents.BaseAgent import BaseAgent


class MentoringAgent(BaseAgent):
    """
    Intelligent mentoring chatbot with two modes:
    1. Session-specific mentoring
    2. Full context mentoring (all sessions)
    Uses SQLite for temporary conversation memory with auto-garbage collection.
    """

    def __init__(
        self,
        llm_client: Any,
        memory_handler: Any,
        config: Optional[Dict] = None
    ):
        super().__init__("MentoringAgent", llm_client)
        
        self.llm_client = llm_client
        self.memory = memory_handler
        self.config = config or {}
        
        # Storage directories
        self.mentoring_dir = Path("learn_files/mentoring")
        self.mentoring_dir.mkdir(exist_ok=True)
        
        # SQLite database for conversation memory
        self.db_path = self.mentoring_dir / "mentoring_memory.db"
        self._init_database()
        
        # Configuration
        self.max_history_tokens = self.config.get("max_history_tokens", 4000)
        self.max_conversation_age_days = self.config.get("max_conversation_age_days", 7)
        self.max_messages_per_session = self.config.get("max_messages_per_session", 50)
        
        # Start garbage collector
        self._schedule_garbage_collection()

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and delegate to appropriate handler."""
        action = input_data.get("action", "chat")
        
        if action == "chat":
            return await self.chat(
                user_id=input_data.get("user_id", ""),
                message=input_data.get("message", ""),
                conversation_id=input_data.get("conversation_id"),
                session_id=input_data.get("session_id"),
                mode=input_data.get("mode", "session")
            )
        elif action == "list_conversations":
            conversations = self.list_conversations(input_data.get("user_id", ""))
            return {"conversations": conversations}
        elif action == "get_conversation_info":
            info = self.get_conversation_info(input_data.get("conversation_id", ""))
            return {"info": info}
        elif action == "get_session_topic":
            topic = self.get_session_topic(input_data.get("session_id", ""))
            return {"topic": topic}
        
        return {"error": f"Unknown action: {action}"}

    def _init_database(self):
        """Initialize SQLite database with proper schema."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Create conversations table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # Create messages table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                token_count INTEGER DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        
        # Create session_content_cache table for fast access
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_content_cache (
                session_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                topics TEXT,
                phases TEXT,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
        
        self.conn.commit()

    def _schedule_garbage_collection(self):
        """Schedule periodic garbage collection."""
        # This will be called on each interaction, not as a background thread
        pass

    def garbage_collect(self):
        """
        Clean up old conversations and messages.
        - Removes conversations older than max_conversation_age_days
        - Truncates conversations with too many messages
        - Removes inactive conversations
        """
        try:
            # Delete old conversations
            cutoff_date = (datetime.now() - timedelta(days=self.max_conversation_age_days)).isoformat()
            self.cursor.execute("""
                DELETE FROM conversations 
                WHERE last_accessed < ? AND is_active = 0
            """, (cutoff_date,))
            
            # Find conversations with too many messages
            self.cursor.execute("""
                SELECT id, message_count FROM conversations 
                WHERE message_count > ?
            """, (self.max_messages_per_session,))
            
            over_limit = self.cursor.fetchall()
            
            for conv_id, count in over_limit:
                # Keep only the most recent messages (last 30)
                keep_count = min(30, self.max_messages_per_session // 2)
                self.cursor.execute("""
                    DELETE FROM messages 
                    WHERE conversation_id = ? 
                    AND id NOT IN (
                        SELECT id FROM messages 
                        WHERE conversation_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    )
                """, (conv_id, conv_id, keep_count))
                
                # Update message count
                self.cursor.execute("""
                    UPDATE conversations 
                    SET message_count = ? 
                    WHERE id = ?
                """, (keep_count, conv_id))
            
            # Delete orphaned messages
            self.cursor.execute("""
                DELETE FROM messages 
                WHERE conversation_id NOT IN (SELECT id FROM conversations)
            """)
            
            self.conn.commit()
            
        except Exception as e:
            print(f"⚠️ Garbage collection error: {e}")

    # ==================== SESSION CONTENT MANAGEMENT ====================

    def _get_session_content(self, session_id: str) -> Dict[str, Any]:
        """
        Get cached session content or load from manifest.
        """
        # Check cache first
        self.cursor.execute("""
            SELECT content, topics, phases FROM session_content_cache 
            WHERE session_id = ?
        """, (session_id,))
        
        result = self.cursor.fetchone()
        
        if result:
            return {
                "content": result[0],
                "topics": json.loads(result[1]) if result[1] else [],
                "phases": json.loads(result[2]) if result[2] else []
            }
        
        # Load from manifest
        manifest_file = Path(f"learn_files/manifests/{session_id}_manifest.json")
        
        if not manifest_file.exists():
            return {"content": "", "topics": [], "phases": []}
        
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            watch_order = manifest.get("watch_order", [])
            topics = [item.get("topic", "") for item in watch_order if item.get("topic")]
            phases = list(set([item.get("phase", "") for item in watch_order if item.get("phase")]))
            
            # Build content summary from all lessons
            content_parts = []
            for item in watch_order[:10]:  # Limit to 10 lessons for context
                text_file = item.get("text_file", "")
                if text_file:
                    try:
                        filepath = Path(text_file)
                        if filepath.exists():
                            with open(filepath, 'r', encoding='utf-8') as f:
                                lesson_content = f.read()
                                # Extract key sections
                                content_parts.append(f"Topic: {item.get('topic', '')}")
                                content_parts.append(lesson_content[:1000])  # Limit per lesson
                    except:
                        pass
            
            full_content = "\n\n".join(content_parts)
            
            # Cache the content
            self.cursor.execute("""
                INSERT OR REPLACE INTO session_content_cache 
                (session_id, content, topics, phases, cached_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                full_content[:50000],  # Limit cached content
                json.dumps(topics[:20]),
                json.dumps(phases[:10]),
                datetime.now().isoformat()
            ))
            self.conn.commit()
            
            return {
                "content": full_content[:50000],
                "topics": topics[:20],
                "phases": phases[:10]
            }
            
        except Exception as e:
            print(f"⚠️ Error loading session content: {e}")
            return {"content": "", "topics": [], "phases": []}

    def _get_all_user_content(self, user_id: str) -> Dict[str, Any]:
        """
        Get all content from all sessions of a user.
        """
        all_topics = []
        all_phases = []
        all_content = []
        
        # Get all sessions for user
        if self.memory:
            sessions = self.memory.get_user_sessions(user_id)
            for session in sessions:
                session_id = session.session_id
                content = self._get_session_content(session_id)
                if content.get("content"):
                    all_content.append(f"Session: {session_id[:20]}...")
                    all_content.append(content["content"])
                    all_topics.extend(content.get("topics", []))
                    all_phases.extend(content.get("phases", []))
        
        return {
            "content": "\n\n".join(all_content)[:100000],  # Limit total content
            "topics": list(set(all_topics))[:30],
            "phases": list(set(all_phases))[:15]
        }

    # ==================== CONVERSATION MANAGEMENT ====================

    def create_conversation(self, user_id: str, session_id: str = None, mode: str = "session") -> str:
        """
        Create a new conversation session.
        """
        # If session_id is None for mode 2, use a special identifier
        if mode == "full" and session_id is None:
            session_id = f"all_sessions_{user_id}"
        
        # Check if an active conversation exists
        if session_id:
            self.cursor.execute("""
                SELECT id FROM conversations 
                WHERE user_id = ? AND session_id = ? AND is_active = 1
                ORDER BY last_accessed DESC LIMIT 1
            """, (user_id, session_id))
            
            result = self.cursor.fetchone()
            if result:
                return str(result[0])
        
        # Create new conversation
        self.cursor.execute("""
            INSERT INTO conversations (session_id, user_id, mode, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, user_id, mode, datetime.now().isoformat(), datetime.now().isoformat()))
        
        self.conn.commit()
        conversation_id = str(self.cursor.lastrowid)
        
        return conversation_id

    def add_message(self, conversation_id: str, role: str, content: str):
        """
        Add a message to a conversation.
        """
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        token_count = len(content) // 4
        
        self.cursor.execute("""
            INSERT INTO messages (conversation_id, role, content, timestamp, token_count)
            VALUES (?, ?, ?, ?, ?)
        """, (conversation_id, role, content, datetime.now().isoformat(), token_count))
        
        # Update conversation
        self.cursor.execute("""
            UPDATE conversations 
            SET last_accessed = ?, message_count = message_count + 1, token_count = token_count + ?
            WHERE id = ?
        """, (datetime.now().isoformat(), token_count, conversation_id))
        
        self.conn.commit()
        
        # Check if we need garbage collection
        self.cursor.execute("SELECT token_count, message_count FROM conversations WHERE id = ?", (conversation_id,))
        result = self.cursor.fetchone()
        
        if result:
            token_count_total = result[0] or 0
            message_count = result[1] or 0
            
            if token_count_total > self.max_history_tokens or message_count > self.max_messages_per_session:
                self._manage_conversation_history(conversation_id)

    def _manage_conversation_history(self, conversation_id: str):
        """
        Manage conversation history by summarizing old messages.
        """
        try:
            # Get all messages
            self.cursor.execute("""
                SELECT id, role, content, timestamp FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
            """, (conversation_id,))
            
            messages = self.cursor.fetchall()
            
            if len(messages) > 20:
                # Keep only the most recent 15 messages
                keep_ids = [m[0] for m in messages[-15:]]
                
                # Delete old messages
                self.cursor.execute("""
                    DELETE FROM messages 
                    WHERE conversation_id = ? AND id NOT IN ({})
                """.format(','.join('?' * len(keep_ids))), [conversation_id] + keep_ids)
                
                # Update message count
                self.cursor.execute("""
                    UPDATE conversations 
                    SET message_count = ? 
                    WHERE id = ?
                """, (len(keep_ids), conversation_id))
                
                self.conn.commit()
        except Exception as e:
            print(f"⚠️ History management error: {e}")

    def get_conversation_history(self, conversation_id: str, limit: int = 20) -> List[Dict]:
        """
        Get conversation history.
        """
        self.cursor.execute("""
            SELECT role, content, timestamp FROM messages 
            WHERE conversation_id = ? 
            ORDER BY timestamp DESC LIMIT ?
        """, (conversation_id, limit))
        
        rows = self.cursor.fetchall()
        
        return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in rows[::-1]]

    def get_conversation_info(self, conversation_id: str) -> Dict:
        """
        Get conversation metadata.
        """
        self.cursor.execute("""
            SELECT id, session_id, user_id, mode, created_at, last_accessed, message_count, token_count
            FROM conversations WHERE id = ?
        """, (conversation_id,))
        
        row = self.cursor.fetchone()
        
        if row:
            return {
                "id": row[0],
                "session_id": row[1],
                "user_id": row[2],
                "mode": row[3],
                "created_at": row[4],
                "last_accessed": row[5],
                "message_count": row[6],
                "token_count": row[7]
            }
        return {}

    def list_conversations(self, user_id: str) -> List[Dict]:
        """
        List all conversations for a user.
        """
        self.cursor.execute("""
            SELECT id, session_id, mode, created_at, last_accessed, message_count
            FROM conversations 
            WHERE user_id = ? AND is_active = 1
            ORDER BY last_accessed DESC
        """, (user_id,))
        
        rows = self.cursor.fetchall()
        
        conversations = []
        for row in rows:
            # Get session topic if available
            session_id = row[1]
            topic = "Unknown"
            if session_id:
                manifest_file = Path(f"learn_files/manifests/{session_id}_manifest.json")
                if manifest_file.exists():
                    try:
                        with open(manifest_file, 'r', encoding='utf-8') as f:
                            manifest = json.load(f)
                            topic = manifest.get("main_topic", "Unknown")
                    except:
                        pass
            
            conversations.append({
                "id": row[0],
                "session_id": row[1],
                "mode": row[2],
                "created_at": row[3],
                "last_accessed": row[4],
                "message_count": row[5],
                "topic": topic
            })
        
        return conversations

    # ==================== CHAT FUNCTIONALITY ====================

    async def chat(
        self,
        user_id: str,
        message: str,
        conversation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        mode: str = "session"
    ) -> Dict[str, Any]:
        """
        Main chat function.
        Mode 1: session-specific
        Mode 2: full context (all sessions)
        """
        # Get or create conversation
        if not conversation_id:
            conversation_id = self.create_conversation(user_id, session_id, mode)
        
        # Get conversation info
        conv_info = self.get_conversation_info(conversation_id)
        
        # Get relevant content based on mode
        if mode == "full":
            # Mode 2: All sessions
            context_data = self._get_all_user_content(user_id)
            context_label = "All your learning sessions"
        else:
            # Mode 1: Specific session
            if not session_id:
                return {"error": "Session ID required for session mode"}
            context_data = self._get_session_content(session_id)
            context_label = f"Session: {session_id[:20]}..."
        
        # Get conversation history
        history = self.get_conversation_history(conversation_id, limit=10)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(
            mode=mode,
            context_label=context_label,
            context_data=context_data,
            history=history
        )
        
        # Build user prompt
        user_prompt = self._build_user_prompt(message, history)
        
        # Get response from LLM
        try:
            response = await self.llm_client.complete(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
        except Exception as e:
            response = f"I apologize, but I encountered an error: {e}. Please try again."

        # Add messages to history
        self.add_message(conversation_id, "user", message)
        self.add_message(conversation_id, "assistant", response)
        
        # Run garbage collection if needed
        self.garbage_collect()
        
        return {
            "response": response,
            "conversation_id": conversation_id,
            "message_count": conv_info.get("message_count", 0) + 2,
            "mode": mode
        }

    def _build_system_prompt(
        self,
        mode: str,
        context_label: str,
        context_data: Dict,
        history: List[Dict]
    ) -> str:
        """
        Build system prompt with context.
        """
        topics = context_data.get("topics", [])
        phases = context_data.get("phases", [])
        content = context_data.get("content", "")
        
        # Summarize content for the prompt
        content_summary = content[:3000] if content else "No specific content available."
        
        prompt = f"""You are Jinvexa Mentor, an intelligent and caring AI learning mentor for the Jinvexa Learning Platform.

Mode: {mode.upper()} - {context_label}

## Your Role
You are a supportive mentor who helps learners understand concepts, answers questions, provides guidance, and encourages learning. You are knowledgeable, patient, and explain things clearly.

## Course Context
Topics: {', '.join(topics[:10])}
Phases: {', '.join(phases[:5])}

Content Summary:
{content_summary}

## Guidelines
1. Be conversational and warm, like a friendly tutor
2. Use examples and analogies to explain concepts
3. Ask clarifying questions when needed
4. Encourage and motivate the learner
5. Reference the course content when relevant
6. If you don't know something, be honest and suggest looking it up

## Style
- Use emojis occasionally for engagement
- Break down complex topics simply
- Celebrate learning progress
- Be patient and encouraging

## Important Notes
- You ONLY have knowledge about the course content provided above
- If asked about something outside this scope, politely redirect to the course
- For coding questions, provide clean, well-explained examples
- Always be helpful and supportive"""
        
        return prompt

    def _build_user_prompt(self, message: str, history: List[Dict]) -> str:
        """
        Build user prompt with conversation history context.
        """
        # Build history context
        history_context = ""
        if history and len(history) > 1:
            recent = history[-5:]  # Last 5 messages for context
            history_context = "\nPrevious conversation:\n"
            for msg in recent:
                role = "Student" if msg["role"] == "user" else "Mentor"
                history_context += f"{role}: {msg['content']}\n"
        
        prompt = f"""{history_context}

Current question/input:
{message}

Please provide a helpful, informative response as the Jinvexa Mentor."""
        
        return prompt

    def get_session_topic(self, session_id: str) -> str:
        """
        Get the topic of a session for display.
        """
        if not session_id:
            return "Unknown"
        
        manifest_file = Path(f"learn_files/manifests/{session_id}_manifest.json")
        if manifest_file.exists():
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    return manifest.get("main_topic", "Unknown")
            except:
                pass
        
        return "Unknown"

    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()