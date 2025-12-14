import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

from src.utils.database import db
from src.utils.logger import logger

class TipsterAnalyzer:
    def __init__(self):
        self.tipster_weights = {}
        self.tipster_specializations = {}
        
    def analyze_tipsters(self) -> Dict[str, Any]:
        """Analyze all tipsters and calculate their accuracy."""
        logger.info("Analyzing tipsters...")
        
        with db.get_session() as session:
            # Get all tipsters with their tips
            tipsters = session.query(db.Tipster).all()
            
            analysis_results = {}
            
            for tipster in tipsters:
                tipster_analysis = self._analyze_tipster(tipster)
                analysis_results[tipster.tipster_id] = tipster_analysis
                
                # Update weights based on recent performance
                self.tipster_weights[tipster.tipster_id] = self._calculate_tipster_weight(tipster_analysis)
                
                # Update specializations
                self.tipster_specializations[tipster.tipster_id] = self._identify_specializations(tipster)
            
            logger.info(f"Analyzed {len(tipsters)} tipsters")
            
            return analysis_results
    
    def _analyze_tipster(self, tipster) -> Dict[str, Any]:
        """Analyze individual tipster performance."""
        # Get tips from last 90 days
        ninety_days_ago = datetime.utcnow() - timedelta(days=90)
        
        with db.get_session() as session:
            recent_tips = session.query(db.Tip).filter(
                db.Tip.tipster_id == tipster.id,
                db.Tip.created_at >= ninety_days_ago,
                db.Tip.is_correct.isnot(None)
            ).all()
            
            if not recent_tips:
                return {
                    'total_tips': 0,
                    'correct_tips': 0,
                    'accuracy': 0.0,
                    'roi': 0.0,
                    'confidence_distribution': {},
                    'performance_by_race_type': {},
                    'recent_trend': 0.0
                }
            
            total_tips = len(recent_tips)
            correct_tips = sum(1 for tip in recent_tips if tip.is_correct)
            accuracy = correct_tips / total_tips
            
            # Calculate ROI (simplified)
            roi = self._calculate_roi(recent_tips)
            
            # Analyze confidence distribution
            confidence_dist = self._analyze_confidence_distribution(recent_tips)
            
            # Analyze performance by race type
            performance_by_type = self._analyze_performance_by_type(tipster, recent_tips)
            
            # Calculate recent trend
            recent_trend = self._calculate_recent_trend(recent_tips)
            
            return {
                'total_tips': total_tips,
                'correct_tips': correct_tips,
                'accuracy': accuracy,
                'roi': roi,
                'confidence_distribution': confidence_dist,
                'performance_by_race_type': performance_by_type,
                'recent_trend': recent_trend
            }
    
    def _calculate_roi(self, tips: List) -> float:
        """Calculate Return on Investment for tipster."""
        # Simplified ROI calculation
        # In reality, this would need actual stake and returns
        
        total_stake = len(tips) * 1.0  # Assume 1 unit per tip
        total_returns = 0.0
        
        for tip in tips:
            if tip.is_correct:
                # Simplified: Assume average odds of 3.0 for winning tips
                total_returns += 3.0
        
        if total_stake > 0:
            roi = ((total_returns - total_stake) / total_stake) * 100
        else:
            roi = 0.0
        
        return roi
    
    def _analyze_confidence_distribution(self, tips: List) -> Dict[str, float]:
        """Analyze how accuracy varies with confidence levels."""
        confidence_bins = {
            'low': (0.0, 0.4),
            'medium': (0.4, 0.7),
            'high': (0.7, 1.0)
        }
        
        results = {}
        
        for bin_name, (lower, upper) in confidence_bins.items():
            bin_tips = [tip for tip in tips if lower <= tip.confidence < upper]
            
            if bin_tips:
                correct = sum(1 for tip in bin_tips if tip.is_correct)
                accuracy = correct / len(bin_tips)
                results[bin_name] = accuracy
            else:
                results[bin_name] = 0.0
        
        return results
    
    def _analyze_performance_by_type(self, tipster, tips: List) -> Dict[str, float]:
        """Analyze tipster performance by race type."""
        performance = {}
        
        # Group tips by race class
        with db.get_session() as session:
            for tip in tips:
                race = session.query(db.Race).filter_by(id=tip.race_id).first()
                if race and race.race_class:
                    race_class = race.race_class
                    
                    if race_class not in performance:
                        performance[race_class] = {'total': 0, 'correct': 0}
                    
                    performance[race_class]['total'] += 1
                    if tip.is_correct:
                        performance[race_class]['correct'] += 1
        
        # Calculate accuracy for each class
        accuracy_by_class = {}
        for race_class, stats in performance.items():
            if stats['total'] > 0:
                accuracy_by_class[race_class] = stats['correct'] / stats['total']
        
        return accuracy_by_class
    
    def _calculate_recent_trend(self, tips: List) -> float:
        """Calculate recent performance trend."""
        if len(tips) < 10:
            return 0.0
        
        # Sort by date
        tips_sorted = sorted(tips, key=lambda x: x.created_at)
        
        # Split into thirds
        third = len(tips_sorted) // 3
        recent = tips_sorted[-third:]
        older = tips_sorted[:third]
        
        # Calculate accuracy for each period
        recent_accuracy = sum(1 for tip in recent if tip.is_correct) / len(recent)
        older_accuracy = sum(1 for tip in older if tip.is_correct) / len(older)
        
        # Calculate trend
        trend = recent_accuracy - older_accuracy
        
        return trend
    
    def _calculate_tipster_weight(self, analysis: Dict[str, Any]) -> float:
        """Calculate weight for tipster based on performance."""
        weight = 0.0
        
        # Base weight on accuracy
        accuracy = analysis['accuracy']
        weight += accuracy * 0.4
        
        # Adjust for sample size
        total_tips = analysis['total_tips']
        sample_size_factor = min(total_tips / 50, 1.0)  # Full weight at 50+ tips
        weight *= sample_size_factor
        
        # Adjust for ROI
        roi = analysis['roi']
        if roi > 0:
            weight *= 1.1
        elif roi < -10:
            weight *= 0.9
        
        # Adjust for recent trend
        trend = analysis['recent_trend']
        if trend > 0.1:
            weight *= 1.2
        elif trend < -0.1:
            weight *= 0.8
        
        # Check confidence calibration
        conf_dist = analysis['confidence_distribution']
        if 'high' in conf_dist and conf_dist['high'] > 0.6:
            weight *= 1.1
        
        return max(0.0, min(1.0, weight))
    
    def _identify_specializations(self, tipster) -> Dict[str, float]:
        """Identify what types of races tipster specializes in."""
        specializations = {}
        
        with db.get_session() as session:
            # Get all tips with results
            tips = session.query(db.Tip).filter_by(tipster_id=tipster.id).all()
            
            for tip in tips:
                race = session.query(db.Race).filter_by(id=tip.race_id).first()
                if race and tip.is_correct is not None:
                    # Analyze by distance
                    if race.distance:
                        dist_range = self._get_distance_range(race.distance)
                        if dist_range not in specializations:
                            specializations[dist_range] = {'total': 0, 'correct': 0}
                        
                        specializations[dist_range]['total'] += 1
                        if tip.is_correct:
                            specializations[dist_range]['correct'] += 1
                    
                    # Analyze by going
                    if race.going and race.going.lower() != 'unknown':
                        going = race.going.lower()
                        if going not in specializations:
                            specializations[going] = {'total': 0, 'correct': 0}
                        
                        specializations[going]['total'] += 1
                        if tip.is_correct:
                            specializations[going]['correct'] += 1
        
        # Calculate specialization scores
        specialization_scores = {}
        for key, stats in specializations.items():
            if stats['total'] >= 5:  # Minimum sample size
                accuracy = stats['correct'] / stats['total']
                specialization_scores[key] = accuracy
        
        return specialization_scores
    
    def _get_distance_range(self, distance: int) -> str:
        """Convert distance to range category."""
        if distance < 1200:
            return 'sprint'
        elif distance < 1600:
            return 'mile'
        elif distance < 2000:
            return 'middle'
        else:
            return 'staying'
    
    def get_most_accurate_tipsters(self, min_tips: int = 20) -> List[Dict[str, Any]]:
        """Get list of most accurate tipsters."""
        analysis = self.analyze_tipsters()
        
        accurate_tipsters = []
        
        for tipster_id, stats in analysis.items():
            if stats['total_tips'] >= min_tips:
                accurate_tipsters.append({
                    'tipster_id': tipster_id,
                    'accuracy': stats['accuracy'],
                    'total_tips': stats['total_tips'],
                    'roi': stats['roi'],
                    'weight': self.tipster_weights.get(tipster_id, 0.0)
                })
        
        # Sort by accuracy
        accurate_tipsters.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return accurate_tipsters
    
    def combine_tipster_predictions(self, race_id: str) -> Dict[str, float]:
        """Combine predictions from all tipsters for a race."""
        with db.get_session() as session:
            race = session.query(db.Race).filter_by(race_id=race_id).first()
            
            if not race:
                return {}
            
            # Get all tips for this race
            tips = session.query(db.Tip).filter_by(race_id=race.id).all()
            
            if not tips:
                return {}
            
            # Group by horse
            horse_predictions = defaultdict(list)
            
            for tip in tips:
                tipster_weight = self.tipster_weights.get(tip.tipster.tipster_id, 0.5)
                
                # Adjust confidence by tipster weight
                adjusted_confidence = tip.confidence * tipster_weight
                
                horse_predictions[tip.horse_name].append({
                    'confidence': adjusted_confidence,
                    'tipster': tip.tipster.tipster_name,
                    'weight': tipster_weight
                })
            
            # Calculate weighted average for each horse
            combined_predictions = {}
            
            for horse, predictions in horse_predictions.items():
                total_weight = sum(p['weight'] for p in predictions)
                weighted_sum = sum(p['confidence'] * p['weight'] for p in predictions)
                
                if total_weight > 0:
                    combined_confidence = weighted_sum / total_weight
                    combined_predictions[horse] = combined_confidence
            
            # Normalize to probabilities
            if combined_predictions:
                total = sum(combined_predictions.values())
                if total > 0:
                    combined_predictions = {k: v/total for k, v in combined_predictions.items()}
            
            return combined_predictions
    
    def get_tipster_recommendations(self, race_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations from top tipsters for a race."""
        # Get most accurate tipsters
        top_tipsters = self.get_most_accurate_tipsters(min_tips=10)[:5]
        
        recommendations = {
            'race_id': race_data.get('race_id'),
            'race_name': race_data.get('race_name'),
            'tipsters': [],
            'consensus_pick': None,
            'value_pick': None
        }
        
        # For each top tipster, simulate what they might pick
        for tipster_info in top_tipsters:
            tipster_id = tipster_info['tipster_id']
            
            # Get tipster specialization
            specialization = self.tipster_specializations.get(tipster_id, {})
            
            # Analyze which horse matches their specialization
            best_horse = self._find_best_horse_for_tipster(race_data, specialization)
            
            if best_horse:
                recommendations['tipsters'].append({
                    'name': tipster_id,
                    'accuracy': tipster_info['accuracy'],
                    'recommended_horse': best_horse['horse_name'],
                    'confidence': best_horse['match_score'],
                    'reasoning': f"Matches specialization in {list(specialization.keys())[:2] if specialization else 'general'}",
                    'weight': tipster_info['weight']
                })
        
        # Find consensus pick
        if recommendations['tipsters']:
            horse_votes = {}
            for tipster_rec in recommendations['tipsters']:
                horse = tipster_rec['recommended_horse']
                weight = tipster_rec['weight']
                
                if horse not in horse_votes:
                    horse_votes[horse] = 0
                horse_votes[horse] += weight
            
            if horse_votes:
                consensus_horse = max(horse_votes.items(), key=lambda x: x[1])
                recommendations['consensus_pick'] = {
                    'horse': consensus_horse[0],
                    'confidence': consensus_horse[1] / len(recommendations['tipsters'])
                }
        
        # Find value pick (horse with good odds but recommended)
        if race_data.get('runners'):
            for runner in race_data['runners']:
                if 'odds' in runner and runner['odds'] > 0:
                    # Check if any tipster recommended this horse
                    tipster_recommended = any(
                        t['recommended_horse'] == runner['horse_name']
                        for t in recommendations['tipsters']
                    )
                    
                    if tipster_recommended and runner['odds'] > 5.0:
                        recommendations['value_pick'] = {
                            'horse': runner['horse_name'],
                            'odds': runner['odds'],
                            'reasoning': 'Good value with tipster support'
                        }
                        break
        
        return recommendations
    
    def _find_best_horse_for_tipster(self, race_data: Dict[str, Any], 
                                   specialization: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Find the best horse in a race for a tipster based on their specialization."""
        if not race_data.get('runners'):
            return None
        
        best_horse = None
        best_score = 0
        
        for runner in race_data['runners']:
            score = 0
            
            # Check distance specialization
            if 'distance' in race_data and 'sprint' in specialization:
                dist_range = self._get_distance_range(race_data['distance'])
                if dist_range in specialization:
                    score += specialization[dist_range] * 0.3
            
            # Check going specialization
            if 'going' in race_data and race_data['going'].lower() in specialization:
                score += specialization[race_data['going'].lower()] * 0.3
            
            # Check form (recent wins)
            if runner.get('career_wins', 0) > 0 and runner.get('career_starts', 1) > 0:
                win_rate = runner['career_wins'] / runner['career_starts']
                score += win_rate * 0.2
            
            # Check jockey/trainer combination
            if runner.get('jockey_name') and runner.get('trainer_name'):
                # In reality, we'd check tipster's historical success with this combo
                score += 0.1
            
            # Check odds (tipsters often like favorites)
            if runner.get('odds', 0) > 0:
                implied_prob = 1 / runner['odds']
                score += implied_prob * 0.1
            
            if score > best_score:
                best_score = score
                best_horse = {
                    'horse_name': runner['horse_name'],
                    'match_score': score,
                    'horse_id': runner.get('horse_id')
                }
        
        if best_horse and best_score > 0:
            best_horse['confidence'] = min(best_score, 1.0)
            return best_horse
        
        return None
    
    def update_tipster_performance(self, race_id: str, results: Dict[str, Any]):
        """Update tipster performance based on race results."""
        with db.get_session() as session:
            race = session.query(db.Race).filter_by(race_id=race_id).first()
            
            if not race:
                return
            
            # Get all tips for this race
            tips = session.query(db.Tip).filter_by(race_id=race.id).all()
            
            winning_horses = [
                results.get('winning_horse_name'),
                results.get('second_horse_name'),
                results.get('third_horse_name')
            ]
            
            for tip in tips:
                # Check if tip was correct (win or place)
                is_correct = tip.horse_name in winning_horses
                
                # Update tip
                tip.is_correct = is_correct
                
                # Update tipster stats
                tipster = tip.tipster
                if is_correct:
                    tipster.winning_tips += 1
                
                tipster.total_tips += 1
                tipster.accuracy_rate = tipster.winning_tips / tipster.total_tips
            
            session.commit()
            logger.info(f"Updated tipster performance for race {race_id}")
    
    def generate_tipster_features(self, race_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate features based on tipster predictions."""
        tipster_features = {}
        
        # Get combined tipster predictions
        combined = self.combine_tipster_predictions(race_data.get('race_id', ''))
        
        if not combined:
            return tipster_features
        
        # For each runner, add tipster-based features
        for runner in race_data.get('runners', []):
            horse_name = runner['horse_name']
            
            if horse_name in combined:
                confidence = combined[horse_name]
                
                # Add features
                runner['tipster_confidence'] = confidence
                runner['tipster_rank'] = sorted(
                    combined.items(), key=lambda x: x[1], reverse=True
                ).index((horse_name, confidence)) + 1
                
                # Is top tipster pick?
                runner['is_top_tipster_pick'] = 1 if runner['tipster_rank'] == 1 else 0
                
                # Tipster consensus strength
                if len(combined) > 1:
                    sorted_confidences = sorted(combined.values(), reverse=True)
                    if len(sorted_confidences) >= 2:
                        runner['tipster_margin'] = sorted_confidences[0] - sorted_confidences[1]
                    else:
                        runner['tipster_margin'] = 0
                else:
                    runner['tipster_margin'] = 0
        
        return tipster_features
