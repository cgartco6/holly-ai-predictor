import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import itertools
from collections import defaultdict
import random

from config import BETTING_CONFIG
from src.utils.logger import logger
from src.models.predictor import HorseRacingPredictor

class PickSix:
    def __init__(self, predictor: HorseRacingPredictor):
        self.predictor = predictor
        self.combinations_generated = 0
        self.historical_results = []
    
    def generate_pick_six_combinations(self, races: List[Dict[str, Any]], 
                                      budget: float = 1000.0) -> Dict[str, Any]:
        """Generate Pick Six combinations."""
        logger.info(f"Generating Pick Six combinations for {len(races)} races")
        
        # Filter races for Pick Six (usually specific races)
        pick_six_races = self._select_pick_six_races(races)
        
        if len(pick_six_races) != 6:
            logger.warning(f"Need exactly 6 races for Pick Six, found {len(pick_six_races)}")
            return {}
        
        combinations = {
            'date': datetime.now().date(),
            'races_included': [r['race_id'] for r in pick_six_races],
            'total_combinations': 0,
            'recommended_combinations': [],
            'budget_required': 0.0,
            'expected_value': 0.0
        }
        
        # Generate predictions for each race
        race_predictions = []
        for race in pick_six_races:
            pred = self.predictor.predict_race(race)
            race_predictions.append(pred)
        
        # Generate combinations
        all_combinations = self._generate_smart_combinations(race_predictions)
        
        # Select combinations within budget
        combinations['recommended_combinations'] = self._select_combinations_within_budget(
            all_combinations, budget
        )
        
        combinations['total_combinations'] = len(all_combinations)
        combinations['budget_required'] = len(combinations['recommended_combinations']) * 2  # R2 per combination
        
        # Calculate expected value
        combinations['expected_value'] = self._calculate_pick_six_ev(combinations['recommended_combinations'])
        
        # Add strategy analysis
        combinations['strategy'] = self._analyze_pick_six_strategy(race_predictions)
        
        logger.info(f"Generated {len(combinations['recommended_combinations'])} "
                   f"Pick Six combinations, budget: R{combinations['budget_required']:.2f}")
        
        return combinations
    
    def _select_pick_six_races(self, races: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select races suitable for Pick Six."""
        # Usually races 3-8 or specific sequence
        # For now, select 6 races with highest prediction confidence
        
        if len(races) < 6:
            return races
        
        # Sort by confidence
        sorted_races = sorted(races, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Try to get a sequence of races
        selected = []
        race_numbers = []
        
        for race in sorted_races:
            if len(selected) < 6:
                # Check if this race is part of a sequence
                race_num = race.get('race_number', 0)
                
                # Prefer sequential races
                if not race_numbers or race_num == max(race_numbers) + 1:
                    selected.append(race)
                    race_numbers.append(race_num)
        
        # If we don't have 6 sequential races, just take top 6
        if len(selected) < 6:
            selected = sorted_races[:6]
        
        return selected
    
    def _generate_smart_combinations(self, race_predictions: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Generate smart combinations for Pick Six."""
        combinations = []
        
        # Get top picks for each race
        top_picks = []
        for pred in race_predictions:
            if pred['predictions']:
                # Take top 3 horses for each race
                top_horses = pred['predictions'][:3]
                top_picks.append(top_horses)
        
        # Generate all combinations of top picks
        # This could be 3^6 = 729 combinations, too many
        # We need to be smarter
        
        # Method 1: Single banker with others
        banker_races = self._select_banker_races(race_predictions)
        
        for banker_idx in banker_races:
            combination = []
            
            for race_idx, horses in enumerate(top_picks):
                if race_idx == banker_idx:
                    # Banker: only pick the top horse
                    combination.append([horses[0]])
                else:
                    # Others: pick top 2 horses
                    combination.append(horses[:2])
            
            # Generate all combinations from this setup
            race_combinations = list(itertools.product(*combination))
            
            # Add to main combinations list
            for combo in race_combinations:
                combinations.append(list(combo))
        
        # Method 2: Focus on races with clear favorites
        clear_favorite_races = []
        for idx, pred in enumerate(race_predictions):
            if pred['predictions']:
                top = pred['predictions'][0]
                second = pred['predictions'][1] if len(pred['predictions']) > 1 else None
                
                if second:
                    margin = top['combined_probability'] - second['combined_probability']
                    if margin > 0.2:  # Clear favorite
                        clear_favorite_races.append(idx)
        
        if len(clear_favorite_races) >= 3:
            # Use clear favorites as bankers
            combination = []
            
            for race_idx, horses in enumerate(top_picks):
                if race_idx in clear_favorite_races:
                    # Clear favorite: only pick top horse
                    combination.append([horses[0]])
                else:
                    # Others: pick top 3 horses
                    combination.append(horses[:3])
            
            race_combinations = list(itertools.product(*combination))
            
            for combo in race_combinations:
                combinations.append(list(combo))
        
        # Remove duplicates
        unique_combinations = []
        seen = set()
        
        for combo in combinations:
            combo_key = tuple((h['horse_name'], h.get('probability', 0)) for h in combo)
            if combo_key not in seen:
                seen.add(combo_key)
                unique_combinations.append(combo)
        
        return unique_combinations
    
    def _select_banker_races(self, race_predictions: List[Dict[str, Any]]) -> List[int]:
        """Select races suitable for banker (single selection)."""
        banker_races = []
        
        for idx, pred in enumerate(race_predictions):
            if pred['predictions']:
                top = pred['predictions'][0]
                
                # Good banker: high probability and clear margin
                if top['combined_probability'] >= 0.7:
                    if len(pred['predictions']) > 1:
                        second = pred['predictions'][1]
                        margin = top['combined_probability'] - second['combined_probability']
                        if margin > 0.15:
                            banker_races.append(idx)
                    else:
                        banker_races.append(idx)
        
        return banker_races
    
    def _select_combinations_within_budget(self, combinations: List[List[Dict[str, Any]]], 
                                         budget: float) -> List[List[Dict[str, Any]]]:
        """Select combinations that fit within budget."""
        # Each combination costs R2
        max_combinations = int(budget / 2)
        
        if len(combinations) <= max_combinations:
            return combinations
        
        # Sort combinations by expected value
        combinations_with_ev = []
        for combo in combinations:
            ev = self._calculate_combination_ev(combo)
            combinations_with_ev.append((combo, ev))
        
        # Sort by expected value
        combinations_with_ev.sort(key=lambda x: x[1], reverse=True)
        
        # Take top combinations within budget
        selected = [combo for combo, ev in combinations_with_ev[:max_combinations]]
        
        return selected
    
    def _calculate_combination_ev(self, combination: List[Dict[str, Any]]) -> float:
        """Calculate expected value of a combination."""
        if not combination:
            return 0.0
        
        # Probability of combination winning
        prob = 1.0
        for horse in combination:
            prob *= horse.get('probability', 0.0)
        
        # Simplified EV calculation
        # In reality, would need Pick Six dividend estimation
        ev = prob * 10000  # Assume R10,000 dividend
        
        return ev
    
    def _calculate_pick_six_ev(self, combinations: List[List[Dict[str, Any]]]) -> float:
        """Calculate total expected value for all combinations."""
        total_ev = 0.0
        
        for combo in combinations:
            total_ev += self._calculate_combination_ev(combo)
        
        # Adjust for cost
        total_cost = len(combinations) * 2  # R2 per combination
        net_ev = total_ev - total_cost
        
        return net_ev
    
    def _analyze_pick_six_strategy(self, race_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and recommend Pick Six strategy."""
        strategy = {
            'recommended_approach': 'balanced',
            'banker_races': [],
            'spread_races': [],
            'risk_level': 'medium',
            'estimated_jackpot': 10000.0  # Default estimate
        }
        
        # Identify banker races
        for idx, pred in enumerate(race_predictions):
            if pred['predictions']:
                top = pred['predictions'][0]
                
                if top['combined_probability'] >= 0.75:
                    strategy['banker_races'].append({
                        'race_number': idx + 1,
                        'horse': top['horse_name'],
                        'probability': top['combined_probability']
                    })
        
        # Identify races to spread (multiple selections)
        for idx, pred in enumerate(race_predictions):
            if idx not in [b['race_number'] - 1 for b in strategy['banker_races']]:
                if pred['predictions']:
                    # Check if race is competitive
                    if len(pred['predictions']) >= 2:
                        top = pred['predictions'][0]
                        second = pred['predictions'][1]
                        margin = top['combined_probability'] - second['combined_probability']
                        
                        if margin < 0.1:  # Very competitive
                            strategy['spread_races'].append({
                                'race_number': idx + 1,
                                'recommended_selections': 3,
                                'top_horses': [h['horse_name'] for h in pred['predictions'][:3]]
                            })
        
        # Determine strategy
        if len(strategy['banker_races']) >= 4:
            strategy['recommended_approach'] = 'banker_heavy'
            strategy['risk_level'] = 'low'
        elif len(strategy['banker_races']) >= 2:
            strategy['recommended_approach'] = 'balanced'
            strategy['risk_level'] = 'medium'
        else:
            strategy['recommended_approach'] = 'spread'
            strategy['risk_level'] = 'high'
        
        return strategy
    
    def simulate_pick_six(self, combinations: Dict[str, Any], 
                         actual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate Pick Six outcome with actual results."""
        simulation = {
            'total_combinations_played': len(combinations.get('recommended_combinations', [])),
            'cost': len(combinations.get('recommended_combinations', [])) * 2,
            'winning_combinations': [],
            'dividends': {},
            'profit_loss': 0.0,
            'jackpot_won': False
        }
        
        # Map race IDs to winning horses
        race_winners = {}
        for result in actual_results:
            race_winners[result['race_id']] = result.get('winning_horse_name')
        
        # Check each combination
        for combo_idx, combo in enumerate(combinations.get('recommended_combinations', [])):
            correct = True
            winning_horses = []
            
            for race_idx, horse in enumerate(combo):
                race_id = combinations['races_included'][race_idx]
                winning_horse = race_winners.get(race_id)
                
                if winning_horse and horse['horse_name'] == winning_horse:
                    winning_horses.append(horse['horse_name'])
                else:
                    correct = False
                    break
            
            if correct:
                simulation['winning_combinations'].append({
                    'combination_number': combo_idx + 1,
                    'winning_horses': winning_horses
                })
        
        # Calculate dividends (simplified)
        if simulation['winning_combinations']:
            # Assume dividend based on number of winners
            if len(simulation['winning_combinations']) == 1:
                dividend = 10000.0  # Jackpot
                simulation['jackpot_won'] = True
            else:
                dividend = 1000.0  # Consolation
            
            simulation['dividends'] = {
                'amount': dividend,
                'winners': len(simulation['winning_combinations'])
            }
            
            simulation['profit_loss'] = dividend - simulation['cost']
        else:
            simulation['profit_loss'] = -simulation['cost']
        
        logger.info(f"Pick Six simulation: {len(simulation['winning_combinations'])} "
                   f"winning combinations, profit/loss: R{simulation['profit_loss']:.2f}")
        
        return simulation
    
    def generate_quick_pick_six(self, races: List[Dict[str, Any]], 
                               num_combinations: int = 5) -> Dict[str, Any]:
        """Generate quick Pick Six combinations for casual players."""
        quick_pick = {
            'date': datetime.now().date(),
            'type': 'quick_pick',
            'combinations': [],
            'total_cost': num_combinations * 2,
            'estimated_jackpot': 10000.0
        }
        
        # Select 6 races
        if len(races) < 6:
            logger.warning(f"Need at least 6 races, found {len(races)}")
            return quick_pick
        
        # Take first 6 races (usually the Pick Six sequence)
        pick_six_races = races[:6]
        
        # Get predictions
        race_predictions = []
        for race in pick_six_races:
            pred = self.predictor.predict_race(race)
            race_predictions.append(pred)
        
        # Generate random combinations
        for _ in range(num_combinations):
            combination = []
            
            for pred in race_predictions:
                if pred['predictions']:
                    # Weighted random selection based on probability
                    horses = pred['predictions'][:4]  # Top 4 horses
                    probs = [h['combined_probability'] for h in horses]
                    
                    # Normalize probabilities
                    total_prob = sum(probs)
                    if total_prob > 0:
                        probs = [p/total_prob for p in probs]
                        
                        # Random selection
                        selected = np.random.choice(horses, p=probs)
                        combination.append(selected)
            
            if len(combination) == 6:
                quick_pick['combinations'].append(combination)
        
        return quick_pick
    
    def analyze_historical_pick_six(self, days_back: int = 90) -> Dict[str, Any]:
        """Analyze historical Pick Six results."""
        analysis = {
            'analysis_period_days': days_back,
            'total_pick_six_events': 0,
            'average_dividend': 0.0,
            'common_patterns': {},
            'recommendations': []
        }
        
        # Load historical results from database
        # This is a simplified version
        # In reality, you'd query actual Pick Six results
        
        # Simulate analysis
        analysis['average_dividend'] = 8500.0
        analysis['jackpot_frequency'] = 'weekly'
        
        # Common patterns (simulated)
        analysis['common_patterns'] = {
            'favorites_included': 4.2,  # Average number of favorites in winning combos
            'longshot_winners': 1.1,    # Average number of longshots (>10/1)
            'most_common_race_for_upset': 5  # Race 5 most commonly has upset
        }
        
        # Recommendations based on analysis
        analysis['recommendations'] = [
            "Include at least 4 favorites in your combination",
            "Take a stand against the favorite in Race 5",
            "Spread in races with large fields (>12 runners)",
            "Use at least one longshot (>10/1) in your combination"
        ]
        
        return analysis
    
    def optimize_for_jackpot(self, races: List[Dict[str, Any]], 
                           carryover_amount: float = 0.0) -> Dict[str, Any]:
        """Optimize Pick Six strategy when there's a jackpot carryover."""
        optimized = {
            'carryover_amount': carryover_amount,
            'recommended_approach': 'aggressive',
            'additional_combinations': 0,
            'estimated_jackpot': carryover_amount * 1.5,  # Estimate
            'special_considerations': []
        }
        
        if carryover_amount > 100000:  # Large carryover
            optimized['recommended_approach'] = 'very_aggressive'
            optimized['additional_combinations'] = 10
            optimized['special_considerations'].append(
                "Large carryover - spread more in competitive races"
            )
            optimized['special_considerations'].append(
                "Consider including more longshots than usual"
            )
        
        elif carryover_amount > 50000:  # Medium carryover
            optimized['recommended_approach'] = 'aggressive'
            optimized['additional_combinations'] = 5
            optimized['special_considerations'].append(
                "Medium carryover - slightly more aggressive approach"
            )
        
        else:  # Small or no carryover
            optimized['recommended_approach'] = 'standard'
            optimized['additional_combinations'] = 0
        
        # Generate combinations with optimized approach
        base_combinations = self.generate_pick_six_combinations(races, budget=1000)
        
        # Add additional combinations if needed
        if optimized['additional_combinations'] > 0:
            # Generate extra combinations
            extra_combinations = self._generate_extra_combinations(
                races, optimized['additional_combinations']
            )
            
            if 'recommended_combinations' in base_combinations:
                base_combinations['recommended_combinations'].extend(extra_combinations)
                base_combinations['budget_required'] += len(extra_combinations) * 2
        
        optimized['combinations'] = base_combinations
        
        return optimized
    
    def _generate_extra_combinations(self, races: List[Dict[str, Any]], 
                                   num_extra: int) -> List[List[Dict[str, Any]]]:
        """Generate extra combinations for jackpot carryover situations."""
        extra_combinations = []
        
        # Take Pick Six races
        pick_six_races = races[:6] if len(races) >= 6 else races
        
        # Get predictions
        race_predictions = []
        for race in pick_six_races:
            pred = self.predictor.predict_race(race)
            race_predictions.append(pred)
        
        # Generate combinations with more longshots
        for _ in range(num_extra):
            combination = []
            
            for pred in race_predictions:
                if pred['predictions']:
                    # Include some longshots
                    # Take horses ranked 3-6 (longshots) 30% of the time
                    if random.random() < 0.3 and len(pred['predictions']) > 5:
                        # Pick a longshot
                        longshot_idx = random.randint(3, min(5, len(pred['predictions'])-1))
                        selected = pred['predictions'][longshot_idx]
                    else:
                        # Pick from top 2
                        top_idx = random.randint(0, min(1, len(pred['predictions'])-1))
                        selected = pred['predictions'][top_idx]
                    
                    combination.append(selected)
            
            if len(combination) == 6:
                extra_combinations.append(combination)
        
        return extra_combinations
