"""
Database Handler for Horse Racing Predictions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.orm import Session
import json
from contextlib import contextmanager

from config import DB_CONFIG
from src.utils.logger import logger

Base = declarative_base()

class Race(Base):
    __tablename__ = 'races'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(String(50), unique=True, nullable=False, index=True)
    meeting_id = Column(String(50), nullable=False, index=True)
    venue = Column(String(100))
    race_date = Column(DateTime, nullable=False, index=True)
    race_time = Column(DateTime, nullable=False)
    race_number = Column(Integer, nullable=False)
    race_name = Column(String(200))
    race_class = Column(String(50))
    distance = Column(Integer)
    going = Column(String(50))
    prize_money = Column(Float)
    track_condition = Column(String(50))
    weather = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    runners = relationship("Runner", back_populates="race", cascade="all, delete-orphan")
    result = relationship("RaceResult", back_populates="race", uselist=False)

class Runner(Base):
    __tablename__ = 'runners'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey('races.id'), nullable=False, index=True)
    horse_id = Column(String(50), nullable=False, index=True)
    horse_name = Column(String(100), nullable=False)
    saddle_number = Column(Integer)
    jockey_id = Column(String(50), index=True)
    jockey_name = Column(String(100))
    trainer_id = Column(String(50), index=True)
    trainer_name = Column(String(100))
    owner_name = Column(String(100))
    weight = Column(Float)
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
    career_places = Column(Integer)
    course_wins = Column(Integer)
    distance_wins = Column(Integer)
    going_wins = Column(String(500))
    odds = Column(Float)
    tipster_rating = Column(Float)
    tipster_confidence = Column(Float)
    form_string = Column(String(100))
    last_5_starts = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    race = relationship("Race", back_populates="runners")
    past_performances = relationship("PastPerformance", back_populates="runner")

class PastPerformance(Base):
    __tablename__ = 'past_performances'
    
    id = Column(Integer, primary_key=True)
    runner_id = Column(Integer, ForeignKey('runners.id'), nullable=False, index=True)
    race_date = Column(DateTime, nullable=False, index=True)
    race_id = Column(String(50))
    meeting = Column(String(100))
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
    time = Column(Float)
    odds = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    runner = relationship("Runner", back_populates="past_performances")

class RaceResult(Base):
    __tablename__ = 'race_results'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey('races.id'), unique=True, nullable=False, index=True)
    winning_horse_id = Column(String(50))
    winning_horse_name = Column(String(100))
    second_horse_id = Column(String(50))
    second_horse_name = Column(String(100))
    third_horse_id = Column(String(50))
    third_horse_name = Column(String(100))
    fourth_horse_id = Column(String(50))
    fourth_horse_name = Column(String(100))
    winning_time = Column(Float)
    winning_distance = Column(Float)
    dividends = Column(JSON)
    sectionals = Column(JSON)
    comments = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    race = relationship("Race", back_populates="result")

class Tipster(Base):
    __tablename__ = 'tipsters'
    
    id = Column(Integer, primary_key=True)
    tipster_id = Column(String(50), unique=True, nullable=False, index=True)
    tipster_name = Column(String(100), nullable=False)
    tipster_type = Column(String(50))
    accuracy_rate = Column(Float)
    roi = Column(Float)
    total_tips = Column(Integer)
    winning_tips = Column(Integer)
    average_odds = Column(Float)
    speciality = Column(String(100))
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    tips = relationship("Tip", back_populates="tipster")

class Tip(Base):
    __tablename__ = 'tips'
    
    id = Column(Integer, primary_key=True)
    tipster_id = Column(Integer, ForeignKey('tipsters.id'), nullable=False, index=True)
    race_id = Column(Integer, ForeignKey('races.id'), nullable=False, index=True)
    horse_id = Column(String(50), nullable=False)
    horse_name = Column(String(100))
    confidence = Column(Float)
    predicted_position = Column(Integer)
    stake_recommendation = Column(String(50))
    reasoning = Column(Text)
    is_correct = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    tipster = relationship("Tipster", back_populates="tips")
    race = relationship("Race")

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey('races.id'), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    prediction_data = Column(JSON, nullable=False)
    top_pick = Column(String(50))
    top_pick_confidence = Column(Float)
    accuracy = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    race = relationship("Race")

class Bet(Base):
    __tablename__ = 'bets'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey('races.id'), nullable=False, index=True)
    horse_id = Column(String(50), nullable=False)
    horse_name = Column(String(100))
    bet_type = Column(String(50))
    stake = Column(Float)
    odds = Column(Float)
    result = Column(String(50))
    profit_loss = Column(Float)
    placed_at = Column(DateTime, default=datetime.utcnow)
    settled_at = Column(DateTime)
    
    race = relationship("Race")

class SystemPerformance(Base):
    __tablename__ = 'system_performance'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
    total_predictions = Column(Integer)
    correct_predictions = Column(Integer)
    accuracy = Column(Float)
    total_bets = Column(Integer)
    winning_bets = Column(Integer)
    total_stake = Column(Float)
    total_return = Column(Float)
    roi = Column(Float)
    bankroll = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseHandler:
    def __init__(self):
        self.engine = create_engine(DB_CONFIG.connection_string, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.create_tables()
    
    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")
    
    @contextmanager
    def get_session(self) -> Session:
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
    
    # Race operations
    def save_race(self, race_data: Dict[str, Any]) -> Optional[Race]:
        """Save or update race data."""
        with self.get_session() as session:
            race = session.query(Race).filter_by(race_id=race_data['race_id']).first()
            
            if race:
                for key, value in race_data.items():
                    if key != 'race_id' and hasattr(race, key):
                        setattr(race, key, value)
                race.updated_at = datetime.utcnow()
            else:
                race = Race(**race_data)
                session.add(race)
            
            session.flush()
            return race
    
    def get_race(self, race_id: str) -> Optional[Race]:
        """Get race by ID."""
        with self.get_session() as session:
            return session.query(Race).filter_by(race_id=race_id).first()
    
    def get_races_by_date(self, date: datetime) -> List[Race]:
        """Get all races for a specific date."""
        with self.get_session() as session:
            start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
            
            return session.query(Race).filter(
                Race.race_date >= start_date,
                Race.race_date < end_date
            ).order_by(Race.race_time).all()
    
    # Runner operations
    def save_runner(self, race_id: int, runner_data: Dict[str, Any]) -> Optional[Runner]:
        """Save or update runner data."""
        with self.get_session() as session:
            runner = session.query(Runner).filter_by(
                race_id=race_id,
                horse_id=runner_data['horse_id']
            ).first()
            
            if runner:
                for key, value in runner_data.items():
                    if key not in ['race_id', 'horse_id'] and hasattr(runner, key):
                        setattr(runner, key, value)
            else:
                runner = Runner(race_id=race_id, **runner_data)
                session.add(runner)
            
            session.flush()
            return runner
    
    def get_runners_for_race(self, race_id: int) -> List[Runner]:
        """Get all runners for a race."""
        with self.get_session() as session:
            return session.query(Runner).filter_by(race_id=race_id).order_by(Runner.saddle_number).all()
    
    # Result operations
    def save_result(self, race_id: int, result_data: Dict[str, Any]) -> Optional[RaceResult]:
        """Save race result."""
        with self.get_session() as session:
            result = session.query(RaceResult).filter_by(race_id=race_id).first()
            
            if result:
                for key, value in result_data.items():
                    if key != 'race_id' and hasattr(result, key):
                        setattr(result, key, value)
            else:
                result = RaceResult(race_id=race_id, **result_data)
                session.add(result)
            
            session.flush()
            
            # Update race with result information
            race = session.query(Race).filter_by(id=race_id).first()
            if race and 'going' in result_data:
                race.going = result_data['going']
            
            return result
    
    # Tip operations
    def save_tip(self, tip_data: Dict[str, Any]) -> Optional[Tip]:
        """Save tip from tipster."""
        with self.get_session() as session:
            tip = Tip(**tip_data)
            session.add(tip)
            session.flush()
            return tip
    
    # Prediction operations
    def save_prediction(self, prediction_data: Dict[str, Any]) -> Optional[Prediction]:
        """Save model prediction."""
        with self.get_session() as session:
            prediction = Prediction(**prediction_data)
            session.add(prediction)
            session.flush()
            return prediction
    
    def get_recent_predictions(self, limit: int = 100) -> List[Prediction]:
        """Get recent predictions."""
        with self.get_session() as session:
            return session.query(Prediction).order_by(
                Prediction.created_at.desc()
            ).limit(limit).all()
    
    # Data export for training
    def export_training_data(self, days_back: int = 365) -> pd.DataFrame:
        """Export historical data for model training."""
        query = """
        SELECT 
            r.*,
            rr.winning_horse_id,
            CASE 
                WHEN rr.winning_horse_id = r.horse_id THEN 1 
                ELSE 0 
            END as is_winner,
            rc.distance as race_distance,
            rc.going as race_going,
            rc.race_class,
            rc.prize_money,
            rc.race_date
        FROM runners r
        JOIN races rc ON r.race_id = rc.id
        LEFT JOIN race_results rr ON rc.id = rr.race_id
        WHERE rc.race_date >= DATE('now', '-' || :days_back || ' days')
        AND rr.winning_horse_id IS NOT NULL
        """
        
        with self.get_session() as session:
            df = pd.read_sql_query(query, session.bind, params={'days_back': days_back})
        
        return df
    
    # Statistics and analytics
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        with self.get_session() as session:
            # Total predictions
            total_preds = session.query(func.count(Prediction.id)).scalar() or 0
            
            # Recent accuracy
            recent_preds = session.query(Prediction).filter(
                Prediction.created_at >= datetime.utcnow() - timedelta(days=30)
            ).all()
            
            if recent_preds:
                recent_accuracy = sum(1 for p in recent_preds if p.accuracy and p.accuracy > 0.5) / len(recent_preds)
            else:
                recent_accuracy = 0.0
            
            # Total races
            total_races = session.query(func.count(Race.id)).scalar() or 0
            
            # Total runners
            total_runners = session.query(func.count(Runner.id)).scalar() or 0
            
            # Recent tipster accuracy
            recent_tips = session.query(Tip).filter(
                Tip.created_at >= datetime.utcnow() - timedelta(days=30),
                Tip.is_correct.isnot(None)
            ).all()
            
            if recent_tips:
                tipster_accuracy = sum(1 for t in recent_tips if t.is_correct) / len(recent_tips)
            else:
                tipster_accuracy = 0.0
            
            return {
                'total_predictions': total_preds,
                'recent_accuracy': recent_accuracy,
                'total_races': total_races,
                'total_runners': total_runners,
                'tipster_accuracy': tipster_accuracy,
                'database_size_mb': self.get_database_size()
            }
    
    def get_database_size(self) -> float:
        """Get database size in MB."""
        with self.get_session() as session:
            if DB_CONFIG.DB_TYPE == 'sqlite':
                result = session.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                size_bytes = result.fetchone()[0]
            else:
                # PostgreSQL
                result = session.execute("SELECT pg_database_size(current_database())")
                size_bytes = result.fetchone()[0]
            
            return size_bytes / (1024 * 1024)  # Convert to MB
    
    # Data cleanup
    def cleanup_old_data(self, days_to_keep: int = 730):
        """Clean up data older than specified days."""
        with self.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Delete old predictions
            old_preds = session.query(Prediction).join(Race).filter(
                Race.race_date < cutoff_date
            ).all()
            
            for pred in old_preds:
                session.delete(pred)
            
            logger.info(f"Cleaned up {len(old_preds)} old predictions")
    
    # Batch operations
    def batch_insert_races(self, races_data: List[Dict[str, Any]]):
        """Insert multiple races in batch."""
        with self.get_session() as session:
            for race_data in races_data:
                race = Race(**race_data)
                session.add(race)
    
    def batch_insert_runners(self, runners_data: List[Dict[str, Any]]):
        """Insert multiple runners in batch."""
        with self.get_session() as session:
            for runner_data in runners_data:
                runner = Runner(**runner_data)
                session.add(runner)
    
    # Advanced queries
    def get_horse_performance(self, horse_id: str, limit: int = 20) -> pd.DataFrame:
        """Get performance history for a horse."""
        query = """
        SELECT 
            rc.race_date,
            rc.race_name,
            rc.distance,
            rc.going,
            rc.race_class,
            r.position,
            r.total_runners,
            r.weight,
            r.jockey_name,
            r.odds,
            rr.winning_horse_id,
            CASE 
                WHEN rr.winning_horse_id = r.horse_id THEN 1 
                ELSE 0 
            END as won
        FROM runners r
        JOIN races rc ON r.race_id = rc.id
        LEFT JOIN race_results rr ON rc.id = rr.race_id
        WHERE r.horse_id = :horse_id
        ORDER BY rc.race_date DESC
        LIMIT :limit
        """
        
        with self.get_session() as session:
            df = pd.read_sql_query(query, session.bind, 
                                 params={'horse_id': horse_id, 'limit': limit})
        
        return df
    
    def get_jockey_statistics(self, jockey_id: str) -> Dict[str, Any]:
        """Get statistics for a jockey."""
        with self.get_session() as session:
            # Total rides
            total_rides = session.query(func.count(Runner.id)).filter_by(
                jockey_id=jockey_id
            ).scalar() or 0
            
            # Wins
            wins = session.query(func.count(Runner.id)).join(
                RaceResult, Runner.race_id == RaceResult.race_id
            ).filter(
                Runner.jockey_id == jockey_id,
                Runner.horse_id == RaceResult.winning_horse_id
            ).scalar() or 0
            
            # Places (1st-3rd)
            places = session.query(func.count(Runner.id)).join(
                RaceResult, Runner.race_id == RaceResult.race_id
            ).filter(
                Runner.jockey_id == jockey_id,
                Runner.horse_id.in_([
                    RaceResult.winning_horse_id,
                    RaceResult.second_horse_id,
                    RaceResult.third_horse_id
                ])
            ).scalar() or 0
            
            # Recent form (last 30 days)
            recent_rides = session.query(Runner).join(Race).filter(
                Runner.jockey_id == jockey_id,
                Race.race_date >= datetime.utcnow() - timedelta(days=30)
            ).count() or 0
            
            recent_wins = session.query(Runner).join(Race).join(RaceResult).filter(
                Runner.jockey_id == jockey_id,
                Race.race_date >= datetime.utcnow() - timedelta(days=30),
                Runner.horse_id == RaceResult.winning_horse_id
            ).count() or 0
            
            return {
                'total_rides': total_rides,
                'wins': wins,
                'places': places,
                'win_percentage': (wins / total_rides * 100) if total_rides > 0 else 0,
                'place_percentage': (places / total_rides * 100) if total_rides > 0 else 0,
                'recent_rides': recent_rides,
                'recent_wins': recent_wins,
                'recent_win_percentage': (recent_wins / recent_rides * 100) if recent_rides > 0 else 0
            }
    
    def get_trainer_statistics(self, trainer_id: str) -> Dict[str, Any]:
        """Get statistics for a trainer."""
        with self.get_session() as session:
            # Total runners
            total_runners = session.query(func.count(Runner.id)).filter_by(
                trainer_id=trainer_id
            ).scalar() or 0
            
            # Wins
            wins = session.query(func.count(Runner.id)).join(
                RaceResult, Runner.race_id == RaceResult.race_id
            ).filter(
                Runner.trainer_id == trainer_id,
                Runner.horse_id == RaceResult.winning_horse_id
            ).scalar() or 0
            
            # Places
            places = session.query(func.count(Runner.id)).join(
                RaceResult, Runner.race_id == RaceResult.race_id
            ).filter(
                Runner.trainer_id == trainer_id,
                Runner.horse_id.in_([
                    RaceResult.winning_horse_id,
                    RaceResult.second_horse_id,
                    RaceResult.third_horse_id
                ])
            ).scalar() or 0
            
            return {
                'total_runners': total_runners,
                'wins': wins,
                'places': places,
                'win_percentage': (wins / total_runners * 100) if total_runners > 0 else 0,
                'place_percentage': (places / total_runners * 100) if total_runners > 0 else 0
            }
    
    def get_track_statistics(self, venue: str) -> Dict[str, Any]:
        """Get statistics for a track/venue."""
        with self.get_session() as session:
            # Total races
            total_races = session.query(func.count(Race.id)).filter_by(
                venue=venue
            ).scalar() or 0
            
            # Favorite winners
            favorite_winners = session.query(func.count(Race.id)).join(
                Runner, Race.id == Runner.race_id
            ).join(
                RaceResult, Race.id == RaceResult.race_id
            ).filter(
                Race.venue == venue,
                Runner.odds == session.query(func.min(Runner.odds)).filter_by(
                    race_id=Race.id
                ).as_scalar(),
                Runner.horse_id == RaceResult.winning_horse_id
            ).scalar() or 0
            
            # Draw advantage
            draw_stats = session.query(
                func.avg(Runner.draw).label('avg_winning_draw'),
                func.stddev(Runner.draw).label('std_winning_draw')
            ).join(
                RaceResult, Runner.race_id == RaceResult.race_id
            ).join(
                Race, Runner.race_id == Race.id
            ).filter(
                Race.venue == venue,
                Runner.horse_id == RaceResult.winning_horse_id
            ).first()
            
            return {
                'total_races': total_races,
                'favorite_win_percentage': (favorite_winners / total_races * 100) if total_races > 0 else 0,
                'avg_winning_draw': draw_stats.avg_winning_draw if draw_stats else 0,
                'std_winning_draw': draw_stats.std_winning_draw if draw_stats else 0
            }

# Singleton instance
db_handler = DatabaseHandler()
