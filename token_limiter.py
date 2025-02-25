import asyncio
import time
from typing import Optional

class TokenLimiter:
    _instance: Optional['TokenLimiter'] = None
    
    def __init__(self, tokens_per_minute: int = 200_000):
        self.tokens_per_minute = tokens_per_minute
        self.tokens = tokens_per_minute
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        
    @classmethod
    def get_instance(cls, tokens_per_minute: int = 200_000) -> 'TokenLimiter':
        """מחזיר מופע יחיד של המחלקה (Singleton)"""
        if cls._instance is None:
            cls._instance = TokenLimiter(tokens_per_minute)
        return cls._instance
    
    async def consume(self, tokens: int) -> None:
        """
        צורך טוקנים וממתין אם אין מספיק
        
        Args:
            tokens: מספר הטוקנים הנדרשים
        """
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            
            # חישוב מחדש של הטוקנים הזמינים
            self.tokens = min(
                self.tokens_per_minute,
                self.tokens + (time_passed * self.tokens_per_minute / 60)
            )
            self.last_update = now
            
            # המתנה אם אין מספיק טוקנים
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) * 60 / self.tokens_per_minute
                await asyncio.sleep(wait_time)
                self.tokens = tokens
            
            self.tokens -= tokens
    
    def estimate_tokens(self, text: str) -> int:
        """
        מעריך את מספר הטוקנים שיידרשו לטקסט
        
        Args:
            text: הטקסט לניתוח
            
        Returns:
            הערכת מספר הטוקנים
        """
        # הערכה מדויקת יותר:
        
        # 1. טוקנים קבועים
        SYSTEM_PROMPT_TOKENS = 250  # הערכה של הפרומפט הקבוע
        JSON_STRUCTURE_TOKENS = 50   # הערכה של מבנה ה-JSON
        
        # 2. טוקנים לטקסט בעברית (בממוצע 4 טוקנים למילה)
        text_tokens = len(text.split()) * 4
        
        # 3. טוקנים לתשובה (בממוצע)
        RESPONSE_TOKENS = 30
        
        total_estimate = (
            SYSTEM_PROMPT_TOKENS +    # פרומפט המערכת
            text_tokens +             # הטקסט עצמו
            JSON_STRUCTURE_TOKENS +   # מבנה ה-JSON
            RESPONSE_TOKENS          # התשובה הצפויה
        )
        
        return total_estimate
    
    async def adjust_tokens(self, estimated: int, actual: int) -> None:
        """
        מתקן את מספר הטוקנים לפי השימוש בפועל
        
        Args:
            estimated: מספר הטוקנים שהוערך
            actual: מספר הטוקנים בפועל
        """
        if actual > estimated:
            await self.consume(actual - estimated)
            print(f"\033[33m[Token Warning] Underestimated: estimated={estimated}, actual={actual}, diff={actual-estimated}\033[0m")
        else:
            self.tokens += estimated - actual
            if (estimated - actual) > 100:  # אם ההערכה גבוהה מדי
                print(f"\033[33m[Token Warning] Overestimated: estimated={estimated}, actual={actual}, diff={estimated-actual}\033[0m")

# יצירת מופע גלובלי לשימוש נוח
limiter = TokenLimiter.get_instance() 