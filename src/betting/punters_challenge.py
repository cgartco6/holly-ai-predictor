import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import itertools
from collections import defaultdict

from config import BETTING_CONFIG
from src.utils.logger import logger
from src.models.predictor import HorseRacingPredictor

class PuntersChallenge:
    def __init__(self, predictor: HorseRacingPredictor):
        self.predictor = predictor
        self.bankroll = BETTING_CONFIG.BANKROLL
        self.betting_history = []
        self.performance_metrics = defaultdict(list)
    
    def generate_challenge_strategy(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate strategy for Punters Challenge."""
        strategy = {
            'date': datetime.now().date(),
            'recommended_selections': [],
            'strategy_type': 'balanced',  # balanced, aggressive, conservative
            'expected_returns': 0.0,
            'risk_level': 'medium'
        }
        
        # Sort races by prediction confidence
        confident_races = sorted(
            predictions,
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        # Select top races (max 8 for challenge)
        selected_races = confident_races[:min(8, len(confident_races))]
        
        for race in selected_races:
            top_pick = race['top_pick']
            value_pick = race.get('value_pick')
            
            selection = {
                'race_id': race['race_id'],
                'race_name': race['race_name'],
                'race_time': race['race_time'],
                'primary_selection': {
                    'horse': top_pick['horse_name'],
                    'probability': top_pick['combined_probability'],
                    'confidence': race['confidence'],
                    'odds': top_pick.get('odds', 0.0)
                }
            }
            
            # Add alternative selection for value
            if value_pick and value_pick['horse_name'] != top_pick['horse_name']:
                selection['alternative_selection'] = {
                    'horse': value_pick['horse_name'],
                    'probability': value_pick['combined_probability'],
                    'value_rating': value_pick.get('value_rating', 'Fair Value'),
                    'odds': value_pick.get('odds', 0.0)
                }
            
            strategy['recommended_selections'].append(selection)
        
        # Calculate expected returns
        strategy['expected_returns'] = self._calculate_expected_returns(strategy)
        
        # Determine strategy type based on confidence
        avg_confidence = np.mean([r['confidence'] for r in selected_races])
        
        if avg_confidence >= 0.8:
            strategy['strategy_type'] = 'aggressive'
            strategy['risk_level'] = 'high'
        elif avg_confidence >= 0.6:
            strategy['strategy_type'] = 'balanced'
            strategy['risk_level'] = 'medium'
        else:
            strategy['strategy_type'] = 'conservative'
            strategy['risk_level'] = 'low'
        
        logger.info(f"Generated Punters Challenge strategy with {len(selected_races)} selections, "
                   f"expected returns: R{strategy['expected_returns']:.2f}")
        
        return strategy
    
    def _calculate_expected_returns(self, strategy: Dict[str, Any]) -> float:
        """Calculate expected returns for the strategy."""
        expected_returns = 0.0
        
        for selection in strategy['recommended_selections']:
            primary = selection['primary_selection']
            probability = primary['probability']
            odds = primary.get('odds', 0.0)
            
            if odds > 0:
                # Expected value calculation
                ev = probability * (odds - 1) - (1 - probability)
                expected_returns += max(ev, 0) * 100  # Scale for visualization
        
        return expected_returns
    
    def simulate_challenge(self, strategy: Dict[str, Any], 
                          actual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate Punters Challenge with actual results."""
        simulation = {
            'strategy_applied': strategy['strategy_type'],
            'selections_made': len(strategy['recommended_selections']),
            'correct_selections': 0,
            'total_points': 0,
            'profit_loss': 0.0,
            'success': False
        }
        
        points_system = {
            1: 3,  # Win
            2: 2,  # Place
            3: 1   # Show
        }
        
        for selection in strategy['recommended_selections']:
            race_id = selection['race_id']
            selected_horse = selection['primary_selection']['horse']
            
            # Find actual result for this race
            result = next((r for r in actual_results if r['race_id'] == race_id), None)
            
            if result:
                winning_horses = [
                    result.get('winning_horse_name'),
                    result.get('second_horse_name'),
                    result.get('third_horse_name')
                ]
                
                # Check if selection finished in top 3
                for position, horse in enumerate(winning_horses, 1):
                    if horse == selected_horse:
                        simulation['correct_selections'] += 1
                        simulation['total_points'] += points_system.get(position, 0)
                        
                        # Calculate profit/loss
                        odds = selection['primary_selection'].get('odds', 0.0)
                        if odds > 0 and position == 1:
                            simulation['profit_loss'] += (odds - 1) * 10  # Assume R10 bet
                        else:
                            simulation['profit_loss'] -= 10  # Lost bet
                        
                        break
                else:
                    simulation['profit_loss'] -= 10  # Lost bet
        
        # Determine if challenge was successful (usually need 6+ correct)
        simulation['success'] = simulation['correct_selections'] >= 6
        
        logger.info(f"Punters Challenge simulation: {simulation['correct_selections']}/"
                   f"{simulation['selections_made']} correct, {simulation['total_points']} points")
        
        return simulation
    
    def optimize_challenge_selections(self, predictions: List[Dict[str, Any]], 
                                    budget: float = 1000.0) -> Dict[str, Any]:
        """Optimize selections for maximum expected value within budget."""
        optimized = {
            'optimized_selections': [],
            'total_expected_value': 0.0,
            'total_budget_used': 0.0,
            'budget_allocation': {}
        }
        
        # Create selection options with expected value
        options = []
        
        for race in predictions:
            top_pick = race['top_pick']
            value_pick = race.get('value_pick')
            
            if top_pick.get('odds', 0) > 0:
                # Calculate expected value for top pick
                ev_top = self._calculate_expected_value(
                    top_pick['combined_probability'],
                    top_pick['odds']
                )
                
                options.append({
                    'race_id': race['race_id'],
                    'horse': top_pick['horse_name'],
                    'type': 'top_pick',
                    'probability': top_pick['combined_probability'],
                    'odds': top_pick['odds'],
                    'expected_value': ev_top,
                    'confidence': race['confidence']
                })
            
            if value_pick and value_pick.get('odds', 0) > 0:
                # Calculate expected value for value pick
                ev_value = self._calculate_expected_value(
                    value_pick['combined_probability'],
                    value_pick['odds']
                )
                
                options.append({
                    'race_id': race['race_id'],
                    'horse': value_pick['horse_name'],
                    'type': 'value_pick',
                    'probability': value_pick['combined_probability'],
                    'odds': value_pick['odds'],
                    'expected_value': ev_value,
                    'confidence': race['confidence'],
                    'value_rating': value_pick.get('value_rating', 'Fair Value')
                })
        
        # Sort by expected value
        options.sort(key=lambda x: x['expected_value'], reverse=True)
        
        # Select optimal combination (knapsack problem simplified)
        selected = []
        budget_used = 0.0
        total_ev = 0.0
        
        # Assume equal stake per selection
        stake_per_selection = budget / min(len(options), 10)
        
        for option in options:
            if budget_used + stake_per_selection <= budget:
                selected.append(option)
                budget_used += stake_per_selection
                total_ev += option['expected_value'] * stake_per_selection
        
        optimized['optimized_selections'] = selected
        optimized['total_expected_value'] = total_ev
        optimized['total_budget_used'] = budget_used
        
        # Calculate budget allocation
        for selection in selected:
            race_id = selection['race_id']
            if race_id not in optimized['budget_allocation']:
                optimized['budget_allocation'][race_id] = 0
            optimized['budget_allocation'][race_id] += stake_per_selection
        
        logger.info(f"Optimized {len(selected)} selections with total EV: R{total_ev:.2f}")
        
        return optimized
    
    def _calculate_expected_value(self, probability: float, odds: float) -> float:
        """Calculate expected value of a bet."""
        if odds <= 0:
            return 0.0
        
        ev = (probability * (odds - 1)) - (1 - probability)
        return ev
    
    def generate_daily_challenge(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate daily Punters Challenge picks."""
        if date is None:
            date = datetime.now()
        
        # Get predictions for the day
        predictions = self.predictor.get_daily_predictions(date)
        
        if not predictions:
            logger.warning(f"No predictions available for {date.date()}")
            return {}
        
        # Generate strategy
        strategy = self.generate_challenge_strategy(predictions)
        
        # Add analysis
        strategy['analysis'] = self._analyze_challenge_predictions(predictions)
        strategy['recommended_bankroll'] = self.bankroll * 0.1  # 10% of bankroll
        
        # Generate confidence intervals
        strategy['confidence_intervals'] = self._calculate_confidence_intervals(predictions)
        
        return strategy
    
    def _analyze_challenge_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze predictions for challenge suitability."""
        analysis = {
            'total_races': len(predictions),
            'high_confidence_races': 0,
            'value_opportunities': 0,
            'average_confidence': 0.0,
            'risk_assessment': 'medium'
        }
        
        if not predictions:
            return analysis
        
        # Count high confidence races
        high_conf = sum(1 for p in predictions if p['confidence'] >= 0.7)
        analysis['high_confidence_races'] = high_conf
        
        # Count value opportunities
        value_opps = sum(1 for p in predictions if p.get('value_pick') is not None)
        analysis['value_opportunities'] = value_opps
        
        # Calculate average confidence
        avg_conf = np.mean([p['confidence'] for p in predictions])
        analysis['average_confidence'] = avg_conf
        
        # Risk assessment
        if avg_conf >= 0.8 and high_conf >= 6:
            analysis['risk_assessment'] = 'low'
        elif avg_conf >= 0.6 and high_conf >= 4:
            analysis['risk_assessment'] = 'medium'
        else:
            analysis['risk_assessment'] = 'high'
        
        return analysis
    
    def _calculate_confidence_intervals(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence intervals for predictions."""
        if not predictions:
            return {}
        
        probabilities = []
        for pred in predictions:
            if pred['predictions']:
                probabilities.append(pred['predictions'][0]['combined_probability'])
        
        if not probabilities:
            return {}
        
        mean_prob = np.mean(probabilities)
        std_prob = np.std(probabilities)
        
        return {
            'mean_probability': float(mean_prob),
            'std_deviation': float(std_prob),
            'confidence_95_lower': float(mean_prob - 1.96 * std_prob / np.sqrt(len(probabilities))),
            'confidence_95_upper': float(mean_prob + 1.96 * std_prob / np.sqrt(len(probabilities)))
        }
    
    def track_performance(self, simulation_results: Dict[str, Any]):
        """Track and analyze challenge performance over time."""
        today = datetime.now().date()
        
        # Store performance metrics
        self.performance_metrics[today] = {
            'correct_selections': simulation_results['correct_selections'],
            'total_selections': simulation_results['selections_made'],
            'points': simulation_results['total_points'],
            'profit_loss': simulation_results['profit_loss'],
            'success': simulation_results['success']
        }
        
        # Update bankroll
        self.bankroll += simulation_results['profit_loss']
        
        # Calculate rolling performance
        recent_days = list(self.performance_metrics.keys())[-30:]  # Last 30 days
        
        if recent_days:
            recent_performance = {
                'days': len(recent_days),
                'total_correct': sum(self.performance_metrics[d]['correct_selections'] for d in recent_days),
                'total_selections': sum(self.performance_metrics[d]['total_selections'] for d in recent_days),
                'total_profit': sum(self.performance_metrics[d]['profit_loss'] for d in recent_days),
                'success_rate': sum(1 for d in recent_days if self.performance_metrics[d]['success']) / len(recent_days)
            }
            
            if recent_performance['total_selections'] > 0:
                recent_performance['accuracy'] = recent_performance['total_correct'] / recent_performance['total_selections']
            
            logger.info(f"Performance tracking: {recent_performance['accuracy']:.3f} accuracy over "
                       f"{recent_performance['days']} days, profit: R{recent_performance['total_profit']:.2f}")
            
            return recent_performance
        
        return {}
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.performance_metrics:
            return {'error': 'No performance data available'}
        
        report = {
            'total_days_tracked': len(self.performance_metrics),
            'current_bankroll': self.bankroll,
            'overall_performance': {},
            'monthly_breakdown': {},
            'streaks': {}
        }
        
        # Calculate overall performance
        total_correct = sum(m['correct_selections'] for m in self.performance_metrics.values())
        total_selections = sum(m['total_selections'] for m in self.performance_metrics.values())
        total_profit = sum(m['profit_loss'] for m in self.performance_metrics.values())
        total_success = sum(1 for m in self.performance_metrics.values() if m['success'])
        
        report['overall_performance'] = {
            'accuracy': total_correct / total_selections if total_selections > 0 else 0,
            'total_profit': total_profit,
            'success_rate': total_success / len(self.performance_metrics),
            'roi': (total_profit / (total_selections * 10)) * 100 if total_selections > 0 else 0
        }
        
        # Monthly breakdown
        monthly_data = defaultdict(list)
        for date_str, metrics in self.performance_metrics.items():
            month_key = date_str.strftime('%Y-%m')
            monthly_data[month_key].append(metrics)
        
        for month, metrics_list in monthly_data.items():
            month_correct = sum(m['correct_selections'] for m in metrics_list)
            month_selections = sum(m['total_selections'] for m in metrics_list)
            month_profit = sum(m['profit_loss'] for m in metrics_list)
            
            report['monthly_breakdown'][month] = {
                'accuracy': month_correct / month_selections if month_selections > 0 else 0,
                'profit': month_profit,
                'success_rate': sum(1 for m in metrics_list if m['success']) / len(metrics_list)
            }
        
        # Calculate streaks
        dates_sorted = sorted(self.performance_metrics.keys())
        current_streak = 0
        best_streak = 0
        current_type = None
        
        for date in dates_sorted:
            success = self.performance_metrics[date]['success']
            
            if success:
                if current_type == 'winning':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'winning'
            else:
                if current_type == 'losing':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'losing'
            
            best_streak = max(best_streak, current_streak)
        
        report['streaks'] = {
            'best_winning_streak': best_streak if current_type == 'winning' else current_streak,
            'current_streak_type': current_type,
            'current_streak_length': current_streak
        }
        
        return report
