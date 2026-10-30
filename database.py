"""
PoseCoach AI - SQLite Database Manager
Handles persistence for sessions, posture transition events, and user settings.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "posecoach.db"


def get_db_connection() -> sqlite3.Connection:
    """Create a database connection with dictionary row access."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Initialize SQLite tables and default settings if they do not exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_uuid TEXT UNIQUE NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_duration_seconds REAL DEFAULT 0.0,
                standing_seconds REAL DEFAULT 0.0,
                sitting_seconds REAL DEFAULT 0.0,
                squatting_seconds REAL DEFAULT 0.0,
                bending_seconds REAL DEFAULT 0.0,
                total_transitions INTEGER DEFAULT 0,
                longest_sitting_seconds REAL DEFAULT 0.0,
                longest_bending_seconds REAL DEFAULT 0.0,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Posture events timeline table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS posture_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                posture TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                duration_seconds REAL DEFAULT 0.0,
                avg_confidence REAL DEFAULT 0.0,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        """)

        # User settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                sitting_reminder_minutes INTEGER DEFAULT 30,
                bending_reminder_minutes INTEGER DEFAULT 10,
                reminder_notifications_enabled INTEGER DEFAULT 1,
                reminder_voice_enabled INTEGER DEFAULT 1,
                voice_enabled INTEGER DEFAULT 1,
                speech_rate REAL DEFAULT 1.0,
                selected_voice TEXT DEFAULT '',
                preferred_language TEXT DEFAULT 'en',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ensure default settings row exists
        cursor.execute("""
            INSERT OR IGNORE INTO user_settings (
                id, sitting_reminder_minutes, bending_reminder_minutes,
                reminder_notifications_enabled, reminder_voice_enabled,
                voice_enabled, speech_rate, selected_voice, preferred_language
            ) VALUES (1, 30, 10, 1, 1, 1, 1.0, '', 'en')
        """)

        # Handle any interrupted sessions from prior crashes
        cursor.execute("""
            UPDATE sessions
            SET status = 'interrupted'
            WHERE status = 'active' OR status = 'paused'
        """)

        conn.commit()


# ==================================================
# SESSION MANAGEMENT
# ==================================================

def create_session() -> Dict[str, Any]:
    """Start a new posture monitoring session."""
    session_uuid = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc).isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (
                session_uuid, start_time, status
            ) VALUES (?, ?, 'active')
        """, (session_uuid, start_time))
        session_id = cursor.lastrowid
        conn.commit()

    return {
        "id": session_id,
        "session_uuid": session_uuid,
        "start_time": start_time,
        "status": "active"
    }


def get_session(session_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a single session by its database ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_active_session() -> Optional[Dict[str, Any]]:
    """Retrieve the current active or paused session if one exists."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM sessions
            WHERE status IN ('active', 'paused')
            ORDER BY id DESC LIMIT 1
        """)
        row = cursor.fetchone()
        return dict(row) if row else None


def update_session_status(session_id: int, status: str) -> bool:
    """Update session status ('active', 'paused', 'completed', 'interrupted')."""
    if status not in ("active", "paused", "completed", "interrupted"):
        raise ValueError(f"Invalid status: {status}")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE sessions SET status = ? WHERE id = ?
        """, (status, session_id))
        conn.commit()
        return cursor.rowcount > 0


def add_posture_event(
    session_id: int,
    posture: str,
    start_time: str,
    end_time: str,
    duration_seconds: float,
    avg_confidence: float = 0.0
) -> int:
    """Record a finalized posture interval into the posture_events table."""
    valid_postures = {"Standing", "Sitting", "Squatting", "Bending"}
    if posture not in valid_postures:
        raise ValueError(f"Invalid posture: {posture}")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO posture_events (
                session_id, posture, start_time, end_time,
                duration_seconds, avg_confidence
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id, posture, start_time, end_time,
            max(0.0, float(duration_seconds)),
            max(0.0, float(avg_confidence))
        ))
        event_id = cursor.lastrowid
        conn.commit()
        return event_id


def save_session_summary(
    session_id: int,
    total_duration_seconds: float,
    standing_seconds: float,
    sitting_seconds: float,
    squatting_seconds: float,
    bending_seconds: float,
    total_transitions: int,
    longest_sitting_seconds: float,
    longest_bending_seconds: float,
    status: str = "completed"
) -> bool:
    """Save finalized session totals and mark session status."""
    end_time = datetime.now(timezone.utc).isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE sessions SET
                end_time = ?,
                total_duration_seconds = ?,
                standing_seconds = ?,
                sitting_seconds = ?,
                squatting_seconds = ?,
                bending_seconds = ?,
                total_transitions = ?,
                longest_sitting_seconds = ?,
                longest_bending_seconds = ?,
                status = ?
            WHERE id = ?
        """, (
            end_time,
            max(0.0, float(total_duration_seconds)),
            max(0.0, float(standing_seconds)),
            max(0.0, float(sitting_seconds)),
            max(0.0, float(squatting_seconds)),
            max(0.0, float(bending_seconds)),
            max(0, int(total_transitions)),
            max(0.0, float(longest_sitting_seconds)),
            max(0.0, float(longest_bending_seconds)),
            status,
            session_id
        ))
        conn.commit()
        return cursor.rowcount > 0


def get_session_events(session_id: int) -> List[Dict[str, Any]]:
    """Retrieve all chronological posture events for a specific session."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM posture_events
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_all_sessions(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Retrieve past monitoring sessions ordered by newest first."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM sessions
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        return [dict(row) for row in cursor.fetchall()]


# ==================================================
# ANALYTICS & TRENDS
# ==================================================

def get_analytics_summary() -> Dict[str, Any]:
    """Calculate aggregated posture statistics across all completed sessions."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total_sessions,
                COALESCE(SUM(total_duration_seconds), 0.0) as total_monitoring_seconds,
                COALESCE(SUM(standing_seconds), 0.0) as total_standing_seconds,
                COALESCE(SUM(sitting_seconds), 0.0) as total_sitting_seconds,
                COALESCE(SUM(squatting_seconds), 0.0) as total_squatting_seconds,
                COALESCE(SUM(bending_seconds), 0.0) as total_bending_seconds,
                COALESCE(SUM(total_transitions), 0) as total_transitions,
                COALESCE(MAX(longest_sitting_seconds), 0.0) as max_sitting_streak_seconds,
                COALESCE(MAX(longest_bending_seconds), 0.0) as max_bending_streak_seconds
            FROM sessions
            WHERE total_duration_seconds > 0
        """)
        summary_row = dict(cursor.fetchone())

        # Average posture event duration
        cursor.execute("""
            SELECT
                COALESCE(AVG(duration_seconds), 0.0) as avg_event_duration_seconds,
                COUNT(*) as total_events
            FROM posture_events
            WHERE duration_seconds > 0
        """)
        event_stats = dict(cursor.fetchone())
        summary_row.update(event_stats)

        # Transition matrix / counts between postures
        cursor.execute("""
            SELECT posture, COUNT(*) as count, COALESCE(SUM(duration_seconds), 0.0) as total_duration
            FROM posture_events
            GROUP BY posture
        """)
        posture_breakdown = {row["posture"]: dict(row) for row in cursor.fetchall()}
        summary_row["posture_breakdown"] = posture_breakdown

        return summary_row


def get_daily_trends(limit_days: int = 7) -> List[Dict[str, Any]]:
    """Retrieve daily posture breakdown for trend charts."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                DATE(start_time) as session_date,
                COUNT(*) as sessions_count,
                COALESCE(SUM(total_duration_seconds), 0.0) as total_seconds,
                COALESCE(SUM(standing_seconds), 0.0) as standing_seconds,
                COALESCE(SUM(sitting_seconds), 0.0) as sitting_seconds,
                COALESCE(SUM(squatting_seconds), 0.0) as squatting_seconds,
                COALESCE(SUM(bending_seconds), 0.0) as bending_seconds
            FROM sessions
            WHERE total_duration_seconds > 0
            GROUP BY DATE(start_time)
            ORDER BY DATE(start_time) DESC
            LIMIT ?
        """, (limit_days,))
        rows = [dict(r) for r in cursor.fetchall()]
        return list(reversed(rows))


# ==================================================
# USER SETTINGS
# ==================================================

def get_user_settings() -> Dict[str, Any]:
    """Retrieve user preferences."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_settings WHERE id = 1")
        row = cursor.fetchone()
        return dict(row) if row else {}


def update_user_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Update user preferences."""
    allowed_keys = {
        "sitting_reminder_minutes",
        "bending_reminder_minutes",
        "reminder_notifications_enabled",
        "reminder_voice_enabled",
        "voice_enabled",
        "speech_rate",
        "selected_voice",
        "preferred_language"
    }
    filtered = {k: v for k, v in settings.items() if k in allowed_keys}
    if not filtered:
        return get_user_settings()

    set_clause = ", ".join(f"{k} = ?" for k in filtered.keys())
    values = list(filtered.values())
    values.append(1)  # id = 1

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE user_settings
            SET {set_clause}, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, values)
        conn.commit()

    return get_user_settings()
