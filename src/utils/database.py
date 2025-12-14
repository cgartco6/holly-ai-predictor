import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import func
from datetime import datetime, timedelta
import json
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from config import DB_CONFIG
from .logger import logger

Base = declarative_base()

class Race(Base):
    __tablename__ = 'races'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(String(50), unique=True, nullable=False)
    meeting_id = Column(String(50), nullable=False)
    race_date = Column(DateTime, nullable=False)
    race_time = Column(DateTime, nullable=False)
    race_number = Column(Integer, nullable=False)
    race_name = Column(String(200))
    race_class = Column(String(50))
    distance = Column(Integer)  # in meters
    going = Column(String(50))
    prize_money = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    runners = relationship("Runner", back_populates="race", cascade="all, delete-orphan")
    results = relationship("RaceResult", back_populates="race", uselist=False)

class Runner(Base):
    __tablename__ = 'runners'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey('races.id'), nullable=False)
    horse_id = Column(String(50), nullable=False)
    horse_name = Column(String(100), nullable=False)
    saddle_number = Column(Integer)
    jockey_id = Column(String(50))
    jockey_name = Column(String(100))
    trainer_id = Column(String(50))
    trainer_name = Column(String(100))
    owner_name = Column(String(100))
    weight = Column(Float)  # in kg
    draw = Column(Integer)
    age = Column(Integer)
    colour = Column(String(50))
    sex = Column(String(10))
    sire = Column(String(100))
    dam = Column(String(100))
    form_rating = Column(Integer)
    official_rating = Column(Integer)
    days_since_last_run = Column(Integer)
    career_starts = Column(Integer)
    career_wins = Column(Integer)
    course_wins = Column(Integer)
    distance_wins = Column(Integer)
    going_wins = Column(String(100))  # JSON string of going preferences
    odds = Column(Float)
    tipster_rating = Column(Float)
    tipster_confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    race = relationship("Race", back_populates="runners")
    past_performances = relationship("PastPerformance", back_populates="runner")
    
    def get_going_wins(self) -> Dict[str, int]:
        """Parse going wins from JSON string."""
        if self.going_wins:
            return json.loads(self.going_wins)
        return {}

class PastPerformance(Base):
    __tablename__ = 'past_performances'
    
    id = Column(Integer, primary_key=True)
    runner_id = Column(Integer, ForeignKey('runners.id'), nullable=False)
    race_date = Column(DateTime, nullable=False)
    race_id = Column(String(50))
    position = Column(Integer)
    total_runners = Column(Integer)
    distance = Column(Integer)
    going = Column(String(50))
    weight = Column(Float)
    jockey_name = Column(String(100))
    prize_money = Column(Float)
    beaten_distance = Column(Float)
    race_class = Column(String(50))
    race_name = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    runner = relationship("Runner", back_populates="past_performances")

class RaceResult(Base):
    __tablename__ = 'race_results'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey('races.id'), unique=True, nullable=False)
    winning_horse_id = Column(String(50))
    winning_horse_name = Column(String(100))
    second_horse_id = Column(String(50))
    second_horse_name = Column(String(100))
    third_horse_id = Column(String(50))
    third_horse_name = Column(String(100))
    fourth_horse_id = Column(String(50))
    fourth_horse_name = Column(String(100))
    winning_time = Column(Float)  # in seconds
    winning_distance = Column(Float)  # in lengths
    dividends = Column(JSON)  # JSON of win/place/quartet etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    race = relationship("Race", back_populates="results")

class Tipster(Base):
    __tablename__ = 'tipsters'
    
    id = Column(Integer, primary_key=True)
    tipster_id = Column(String(50), unique=True, nullable=False)
    tipster_name = Column(String(100), nullable=False)
    tipster_type = Column(String(50))  # professional, amateur, system
    accuracy_rate = Column(Float)  # overall accuracy
    roi = Column(Float)  # return on investment
    total_tips = Column(Integer)
    winning_tips = Column(Integer)
    average_odds = Column(Float)
    speciality = Column(String(100))  # sprints, staying, specific tracks
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tips = relationship("Tip", back_populates="tipster")

class Tip(Base):
    __tablename__ = 'tips'
    
    id = Column(Integer, primary_key=True)
    tipster_id = Column(Integer, ForeignKey('tipsters.id'), nullable=False)
    race_id = Column(Integer, ForeignKey('races.id'), nullable=False)
    horse_id = Column(String(50), nullable=False)
    horse_name = Column(String(100))
    confidence = Column(Float)  # 0-1 scale
    predicted_position = Column(Integer)
    stake_recommendation = Column(String(50))  # win, each-way, place
    reasoning = Column(Text)
    is_correct = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    tipster = relationship("Tipster", back_populates="tips")
    race = relationship("Race")

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey('races.id'), nullable=False)
    model_name = Column(String(100), nullable=False)
    prediction_data = Column(JSON, nullable=False)  # JSON of horse_id -> confidence
    top_pick = Column(String(50))
    top_pick_confidence = Column(Float)
    accuracy = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    race = relationship("Race")

class Database:
    def __init__(self):
        self.engine = create_engine(DB_CONFIG.connection_string)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")
    
    @contextmanager
    def get_session(self):
        """Provide a transactional scope around a series of operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def save_race(self, race_data: Dict[str, Any]) -> Optional[Race]:
        """Save or update race data."""
        with self.get_session() as session:
            # Check if race exists
            race = session.query(Race).filter_by(race_id=race_data['race_id']).first()
            
            if race:
                # Update existing race
                for key, value in race_data.items():
                    if key != 'race_id' and hasattr(race, key):
                        setattr(race, key, value)
                race.updated_at = datetime.utcnow()
            else:
                # Create new race
                race = Race(**race_data)
                session.add(race)
            
            session.flush()
            return race
    
    def save_runner(self, race_id: int, runner_data: Dict[str, Any]) -> Optional[Runner]:
        """Save or update runner data."""
        with self.get_session() as session:
            # Check if runner exists for this race
            runner = session.query(Runner).filter_by(
                race_id=race_id,
                horse_id=runner_data['horse_id']
            ).first()
            
            if runner:
                # Update existing runner
                for key, value in runner_data.items():
                    if key not in ['race_id', 'horse_id'] and hasattr(runner, key):
                        setattr(runner, key, value)
            else:
                # Create new runner
                runner = Runner(race_id=race_id, **runner_data)
                session.add(runner)
            
            session.flush()
            return runner
    
    def save_race_result(self, race_id: int, result_data: Dict[str, Any]) -> Optional[RaceResult]:
        """Save race result."""
        with self.get_session() as session:
            # Check if result exists
            result = session.query(RaceResult).filter_by(race_id=race_id).first()
            
            if result:
                # Update existing result
                for key, value in result_data.items():
                    if key != 'race_id' and hasattr(result, key):
                        setattr(result, key, value)
            else:
                # Create new result
                result = RaceResult(race_id=race_id, **result_data)
                session.add(result)
            
            session.flush()
            return result
    
    def save_tip(self, tip_data: Dict[str, Any]) -> Optional[Tip]:
        """Save tip from tipster."""
        with self.get_session() as session:
            # Update tipster stats if tipster exists
            tipster = session.query(Tipster).filter_by(
                tipster_id=tip_data['tipster_id']
            ).first()
            
            if not tipster:
                # Create new tipster
                tipster = Tipster(
                    tipster_id=tip_data['tipster_id'],
                    tipster_name=tip_data.get('tipster_name', 'Unknown'),
                    total_tips=0,
                    winning_tips=0,
                    accuracy_rate=0.0
                )
                session.add(tipster)
            
            # Save tip
            tip = Tip(**tip_data)
            session.add(tip)
            
            session.flush()
            return tip
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> Optional[Prediction]:
        """Save model prediction."""
        with self.get_session() as session:
            prediction = Prediction(**prediction_data)
            session.add(prediction)
            session.flush()
            return prediction
    
    def get_races_for_date(self, date: datetime) -> List[Race]:
        """Get all races for a specific date."""
        with self.get_session() as session:
            start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
            
            races = session.query(Race).filter(
                Race.race_date >= start_date,
                Race.race_date < end_date
            ).order_by(Race.race_time).all()
            
            return races
    
    def get_past_performances(self, horse_id: str, limit: int = 10) -> List[PastPerformance]:
        """Get past performances for a horse."""
        with self.get_session() as session:
            # Find runner IDs for this horse
            runner_ids = session.query(Runner.id).filter_by(horse_id=horse_id).all()
            runner_ids = [r[0] for r in runner_ids]
            
            performances = session.query(PastPerformance).filter(
                PastPerformance.runner_id.in_(runner_ids)
            ).order_by(PastPerformance.race_date.desc()).limit(limit).all()
            
            return performances
    
    def get_tipster_accuracy(self, tipster_id: str) -> Dict[str, Any]:
        """Calculate tipster accuracy statistics."""
        with self.get_session() as session:
            tipster = session.query(Tipster).filter_by(tipster_id=tipster_id).first()
            
            if not tipster:
                return {}
            
            # Get recent tips (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_tips = session.query(Tip).filter(
                Tip.tipster_id == tipster.id,
                Tip.created_at >= thirty_days_ago,
                Tip.is_correct.isnot(None)
            ).all()
            
            if not recent_tips:
                return {
                    'overall_accuracy': tipster.accuracy_rate,
                    'recent_accuracy': 0.0,
                    'total_tips': tipster.total_tips,
                    'winning_tips': tipster.winning_tips
                }
            
            recent_correct = sum(1 for tip in recent_tips if tip.is_correct)
            recent_accuracy = recent_correct / len(recent_tips)
            
            return {
                'overall_accuracy': tipster.accuracy_rate,
                'recent_accuracy': recent_accuracy,
                'total_tips': tipster.total_tips,
                'winning_tips': tipster.winning_tips,
                'roi': tipster.roi,
                'speciality': tipster.speciality
            }
    
    def get_training_data(self, days_back: int = 365) -> pd.DataFrame:
        """Get historical data for model training."""
        with self.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            query = """
            SELECT 
                r.*,
                rr.winning_horse_id,
                CASE WHEN rr.winning_horse_id = r.horse_id THEN 1 ELSE 0 END as is_winner,
                t.accuracy_rate as tipster_accuracy,
                t.speciality as tipster_speciality
            FROM runners r
            JOIN races rc ON r.race_id = rc.id
            LEFT JOIN race_results rr ON rc.id = rr.race_id
            LEFT JOIN tips tp ON r.race_id = tp.race_id AND r.horse_id = tp.horse_id
            LEFT JOIN tipsters t ON tp.tipster_id = t.id
            WHERE rc.race_date >= :cutoff_date
            AND rr.winning_horse_id IS NOT NULL
            """
            
            df = pd.read_sql_query(query, session.bind, params={'cutoff_date': cutoff_date})
            return df
    
    def update_tipster_stats(self, tipster_id: str):
        """Update tipster statistics after new results."""
        with self.get_session() as session:
            tipster = session.query(Tipster).filter_by(tipster_id=tipster_id).first()
            
            if not tipster:
                return
            
            # Count total tips and correct tips
            tips = session.query(Tip).filter_by(tipster_id=tipster.id).all()
            
            total_tips = len(tips)
            correct_tips = sum(1 for tip in tips if tip.is_correct)
            
            if total_tips > 0:
                accuracy = correct_tips / total_tips
                
                # Calculate ROI (simplified)
                # This would need actual betting amounts and returns
                tipster.total_tips = total_tips
                tipster.winning_tips = correct_tips
                tipster.accuracy_rate = accuracy
                tipster.last_updated = datetime.utcnow()
    
    def cleanup_old_data(self, days_to_keep: int = 730):
        """Clean up data older than specified days."""
        with self.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Delete old predictions
            old_predictions = session.query(Prediction).join(Race).filter(
                Race.race_date < cutoff_date
            ).all()
            
            for pred in old_predictions:
                session.delete(pred)
            
            logger.info(f"Cleaned up {len(old_predictions)} old predictions")

# Singleton instance
db = Database()
