# D:\Jinvexa\Agents\MemoryHandler.py

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib
import pickle
from dataclasses import dataclass, asdict, field

from Models.UserProfile import UserProfile
from Models.LearningPlan import LearningPlan


@dataclass
class SessionMemory:
    """Represents a complete session memory"""
    session_id: str
    user_id: str
    mode: str  # "goal" or "reference"
    created_at: str
    last_accessed: str
    conversation_history: List[Dict]
    extracted_data: Optional[Dict] = None
    concepts: Optional[Dict] = None
    knowledge_graph: Optional[Dict] = None
    gap_analysis: Optional[Dict] = None
    learning_plan: Optional[Dict] = None
    user_profile: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "mode": self.mode,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "conversation_history": self.conversation_history,
            "extracted_data": self.extracted_data,
            "concepts": self.concepts,
            "knowledge_graph": self.knowledge_graph,
            "gap_analysis": self.gap_analysis,
            "learning_plan": self.learning_plan,
            "user_profile": self.user_profile,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SessionMemory":
        return cls(**data)


class MemoryHandler:
    """
    Handles persistent memory storage for the Learning Discovery Agent.
    Supports JSON file storage.
    """
    
    def __init__(self, storage_dir: str = "memory_storage", storage_type: str = "json"):
        """
        Initialize the Memory Handler
        
        Args:
            storage_dir: Directory for storing memory files
            storage_type: "json" (default)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_type = storage_type
        
        # Create storage directories
        self.storage_dir.mkdir(exist_ok=True)
        (self.storage_dir / "sessions").mkdir(exist_ok=True)
        (self.storage_dir / "profiles").mkdir(exist_ok=True)
        
        # In-memory cache for fast access
        self._cache: Dict[str, SessionMemory] = {}
        self._profile_cache: Dict[str, UserProfile] = {}
        
        print(f"💾 MemoryHandler initialized at: {self.storage_dir}")
    
    # ==================== SESSION MANAGEMENT ====================
    
    def create_session(
        self,
        user_id: str,
        mode: str,
        user_profile: Optional[UserProfile] = None
    ) -> str:
        """
        Create a new session
        
        Returns:
            session_id: Unique session identifier
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hash_suffix = hashlib.md5(f"{user_id}{datetime.now()}".encode()).hexdigest()[:8]
        session_id = f"{user_id}_{timestamp}_{hash_suffix}"
        
        session = SessionMemory(
            session_id=session_id,
            user_id=user_id,
            mode=mode,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            conversation_history=[],
            user_profile=user_profile.to_dict() if user_profile else None
        )
        
        # Save to cache
        self._cache[session_id] = session
        
        # Save to storage
        self._save_session(session)
        
        print(f"✅ Session created: {session_id[:20]}...")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        """Get a session by ID"""
        # Check cache first
        if session_id in self._cache:
            return self._cache[session_id]
        
        # Check storage
        session = self._load_session(session_id)
        if session:
            self._cache[session_id] = session
        return session
    
    def update_session(self, session: SessionMemory):
        """Update an existing session"""
        session.last_accessed = datetime.now().isoformat()
        
        # Update cache
        self._cache[session.session_id] = session
        
        # Update storage
        self._save_session(session)
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        # Remove from cache
        if session_id in self._cache:
            del self._cache[session_id]
        
        # Remove from storage
        self._delete_session(session_id)
    
    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[SessionMemory]:
        """Get all sessions for a user"""
        sessions = []
        session_dir = self.storage_dir / "sessions"
        
        if session_dir.exists():
            # Get all session files for this user
            for file_path in session_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if data.get("user_id") == user_id:
                        # Parse the data properly
                        parsed_data = self._parse_session_data(data)
                        session = SessionMemory.from_dict(parsed_data)
                        sessions.append(session)
                except Exception as e:
                    print(f"⚠️ Error loading session {file_path}: {e}")
                    continue
        
        # Sort by created_at (newest first) and limit
        sessions.sort(key=lambda x: x.created_at, reverse=True)
        return sessions[:limit]
    
    def _parse_session_data(self, data: Dict) -> Dict:
        """Parse session data, converting JSON strings back to objects"""
        parsed = data.copy()
        
        # Keys that should be parsed as JSON
        json_keys = ["conversation_history", "extracted_data", "concepts", 
                     "knowledge_graph", "gap_analysis", "learning_plan", 
                     "user_profile", "metadata"]
        
        for key in json_keys:
            if key in parsed and parsed[key] is not None and isinstance(parsed[key], str):
                try:
                    parsed[key] = json.loads(parsed[key])
                except:
                    parsed[key] = None
            elif key in parsed and parsed[key] is None:
                parsed[key] = None
        
        return parsed
    
    def add_conversation_message(self, session_id: str, role: str, message: str):
        """Add a message to session conversation history"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.conversation_history.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        self.update_session(session)
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        session = self.get_session(session_id)
        return session.conversation_history if session else []
    
    def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session"""
        session = self.get_session(session_id)
        if session:
            session.conversation_history = []
            self.update_session(session)
    
    # ==================== PROFILE MANAGEMENT ====================
    
    def save_profile(self, profile: UserProfile):
        """Save a user profile"""
        # Save to cache
        self._profile_cache[profile.user_id] = profile
        
        # Save to storage
        self._save_profile(profile)
    
    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load a user profile"""
        # Check cache first
        if user_id in self._profile_cache:
            return self._profile_cache[user_id]
        
        # Check storage
        profile = self._load_profile(user_id)
        if profile:
            self._profile_cache[user_id] = profile
        return profile
    
    def delete_profile(self, user_id: str):
        """Delete a user profile"""
        # Remove from cache
        if user_id in self._profile_cache:
            del self._profile_cache[user_id]
        
        # Remove from storage
        self._delete_profile(user_id)
    
    def list_profiles(self) -> List[str]:
        """List all user IDs with profiles"""
        profile_dir = self.storage_dir / "profiles"
        if profile_dir.exists():
            return [f.stem for f in profile_dir.glob("*.json")]
        return []
    
    # ==================== STATISTICS AND METRICS ====================
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get learning statistics for a user"""
        sessions = self.get_user_sessions(user_id, limit=100)
        
        stats = {
            "user_id": user_id,
            "total_sessions": len(sessions),
            "active_sessions": len([s for s in sessions if s.learning_plan]),
            "total_conversations": sum(len(s.conversation_history) for s in sessions),
            "learning_plans": [],
            "knowledge_progress": {}
        }
        
        # Collect learning plans
        for session in sessions:
            if session.learning_plan:
                plan = session.learning_plan
                stats["learning_plans"].append({
                    "topic": plan.get("main_topic", "Unknown"),
                    "goal": plan.get("goal", "Unknown"),
                    "estimated_hours": plan.get("estimated_time_hours", 0)
                })
        
        # Get knowledge progress from profile
        profile = self.load_profile(user_id)
        if profile:
            stats["knowledge_progress"] = {
                "known_concepts": len(profile.known_concepts),
                "topics": list(profile.known_concepts.keys())[:20]
            }
        
        return stats
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics across all users"""
        profiles = self.list_profiles()
        
        stats = {
            "total_users": len(profiles),
            "total_sessions": 0,
            "total_conversations": 0,
            "learning_plans_generated": 0
        }
        
        # Count sessions from JSON files
        session_dir = self.storage_dir / "sessions"
        if session_dir.exists():
            stats["total_sessions"] = len(list(session_dir.glob("*.json")))
            stats["total_conversations"] = stats["total_sessions"] * 5
        
        return stats
    
    # ==================== STORAGE BACKEND METHODS ====================
    
    def _save_session(self, session: SessionMemory):
        """Save session to JSON storage"""
        data = session.to_dict()
        
        # Convert complex objects to JSON strings for storage
        for key in ["conversation_history", "extracted_data", "concepts", 
                    "knowledge_graph", "gap_analysis", "learning_plan", 
                    "user_profile", "metadata"]:
            if data.get(key) is not None:
                try:
                    data[key] = json.dumps(data[key], ensure_ascii=False)
                except:
                    data[key] = None
            else:
                data[key] = None
        
        # Save to file
        session_file = self.storage_dir / "sessions" / f"{session.session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_session(self, session_id: str) -> Optional[SessionMemory]:
        """Load session from JSON storage"""
        session_file = self.storage_dir / "sessions" / f"{session_id}.json"
        if session_file.exists():
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Parse JSON strings back to objects
                data = self._parse_session_data(data)
                
                return SessionMemory.from_dict(data)
            except Exception as e:
                print(f"⚠️ Error loading session {session_id}: {e}")
        
        return None
    
    def _delete_session(self, session_id: str):
        """Delete session from storage"""
        session_file = self.storage_dir / "sessions" / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
    
    def _save_profile(self, profile: UserProfile):
        """Save profile to JSON storage"""
        data = {
            "user_id": profile.user_id,
            "profile_data": profile.to_dict(),
            "created_at": profile.created_at,
            "updated_at": profile.updated_at
        }
        
        profile_file = self.storage_dir / "profiles" / f"{profile.user_id}.json"
        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load profile from JSON storage"""
        profile_file = self.storage_dir / "profiles" / f"{user_id}.json"
        if profile_file.exists():
            try:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return UserProfile.from_dict(data.get("profile_data", {}))
            except Exception as e:
                print(f"⚠️ Error loading profile {user_id}: {e}")
        return None
    
    def _delete_profile(self, user_id: str):
        """Delete profile from storage"""
        profile_file = self.storage_dir / "profiles" / f"{user_id}.json"
        if profile_file.exists():
            profile_file.unlink()
    
    # ==================== UTILITY METHODS ====================
    
    def clear_cache(self):
        """Clear in-memory cache"""
        self._cache.clear()
        self._profile_cache.clear()
    
    def cleanup_old_sessions(self, days: int = 30):
        """Delete sessions older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        session_dir = self.storage_dir / "sessions"
        
        if session_dir.exists():
            for file_path in session_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    created_at = datetime.fromisoformat(data.get("created_at", ""))
                    if created_at < cutoff:
                        file_path.unlink()
                except:
                    continue
    
    def export_data(self, user_id: str, export_dir: str = "exports") -> str:
        """Export all data for a user"""
        export_path = Path(export_dir) / user_id
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export profile
        profile = self.load_profile(user_id)
        if profile:
            with open(export_path / "profile.json", 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
        
        # Export sessions
        sessions = self.get_user_sessions(user_id)
        sessions_data = [s.to_dict() for s in sessions]
        with open(export_path / "sessions.json", 'w') as f:
            json.dump(sessions_data, f, indent=2)
        
        return str(export_path)