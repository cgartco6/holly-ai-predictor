import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class ResultsCollector:
    def __init__(self, data_dir="data/raw/results"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def generate_sample_results(self, num_races=50):
        """Generate realistic race results"""
        np.random.seed(42)
        
        all_results = []
        
        for race_id in range(1, num_races + 1):
            race_date = datetime.now() - timedelta(days=np.random.randint(1, 60))
            
            # Generate race result
            race_result = {
                'race_id': f'R{race_id:04d}',
                'race_date': race_date.strftime('%Y-%m-%d'),
                'track': np.random.choice(['Churchill Downs', 'Santa Anita', 'Belmont Park']),
                'race_number': race_id % 10 + 1,
                'distance': np.random.choice([5.0, 6.0, 7.0, 8.0, 9.0]),
                'surface': np.random.choice(['Dirt', 'Turf']),
                'track_condition': np.random.choice(['Fast', 'Good', 'Sloppy']),
                'purse': np.random.choice([20000, 50000, 100000]),
                'winning_time': self._generate_winning_time(),
                'final_odds': {},
                'results': [],
                'payouts': {}
            }
            
            # Generate horse results
            num_horses = np.random.randint(6, 12)
            finish_positions = list(range(1, num_horses + 1))
            np.random.shuffle(finish_positions)
            
            for horse_num in range(1, num_horses + 1):
                horse_result = self._generate_horse_result(
                    horse_num, finish_positions[horse_num-1], race_result
                )
                race_result['results'].append(horse_result)
                
                # Store final odds
                race_result['final_odds'][horse_result['horse_name']] = horse_result['final_odds']
            
            # Generate payouts
            race_result['payouts'] = self._generate_payouts(race_result['results'])
            
            all_results.append(race_result)
        
        # Save to file
        filename = f"{self.data_dir}/results_{datetime.now().strftime('%Y%m')}.json"
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Generated {len(all_results)} race results to {filename}")
        return all_results
    
    def _generate_winning_time(self):
        """Generate realistic winning time based on distance"""
        distance = np.random.choice([5.0, 6.0, 7.0, 8.0, 9.0])
        
        # Base times for 6 furlongs
        if distance == 5.0:
            return f"0:{np.random.randint(56, 60)}.{np.random.randint(0, 100):02d}"
        elif distance == 6.0:
            return f"1:{np.random.randint(9, 13)}.{np.random.randint(0, 100):02d}"
        elif distance == 7.0:
            return f"1:{np.random.randint(21, 25)}.{np.random.randint(0, 100):02d}"
        elif distance == 8.0:
            return f"1:{np.random.randint(34, 38)}.{np.random.randint(0, 100):02d}"
        else:  # 9.0
            return f"1:{np.random.randint(48, 52)}.{np.random.randint(0, 100):02d}"
    
    def _generate_horse_result(self, horse_num, finish_position, race_info):
        """Generate result for individual horse"""
        horse_names = [
            'Midnight Thunder', 'Silver Streak', 'Golden Crown', 'Red Rocket',
            'Blue Diamond', 'Green Meadow', 'Black Shadow', 'White Knight'
        ]
        
        jockeys = ['John Velazquez', 'Mike Smith', 'Javier Castellano', 'Irad Ortiz Jr']
        
        horse = {
            'horse_id': f'H{race_info["race_id"][1:]}{horse_num:02d}',
            'program_number': horse_num,
            'horse_name': np.random.choice(horse_names) + f' {horse_num}',
            'jockey': np.random.choice(jockeys),
            'trainer': np.random.choice(['Bob Baffert', 'Todd Pletcher', 'Chad Brown']),
            'finish_position': finish_position,
            'final_odds': round(np.random.uniform(1.5, 25.0), 2),
            'win_pool': np.random.randint(5000, 50000),
            'place_pool': np.random.randint(3000, 30000),
            'show_pool': np.random.randint(2000, 20000),
            'weight': np.random.randint(1150, 1220),
            'claimed': np.random.choice([True, False], p=[0.2, 0.8]),
            'claim_price': np.random.choice([0, 15000, 25000, 40000]),
            'scratched': False,
            'disqualified': False
        }
        
        # Add beaten lengths and times
        if finish_position == 1:
            horse['beaten_lengths'] = 0
            horse['winning_margin'] = round(np.random.uniform(0.5, 4.0), 1)
        else:
            horse['beaten_lengths'] = round(np.random.uniform(0.5, 15.0), 1)
        
        # Add pace information
        horse['pace_data'] = {
            'fractional_times': self._generate_fractional_times(race_info['distance']),
            'position_call': {
                'start': np.random.randint(1, horse['finish_position'] + 3),
                'quarter': np.random.randint(1, horse['finish_position'] + 2),
                'half': np.random.randint(1, horse['finish_position'] + 1),
                'stretch': horse['finish_position']
            }
        }
        
        return horse
    
    def _generate_fractional_times(self, distance):
        """Generate fractional times for race"""
        if distance == 6.0:
            return {
                'quarter': f"0:{np.random.randint(21, 24)}.{np.random.randint(0, 100):02d}",
                'half': f"0:{np.random.randint(44, 47)}.{np.random.randint(0, 100):02d}",
                'three_quarters': f"1:{np.random.randint(9, 12)}.{np.random.randint(0, 100):02d}"
            }
        return {}
    
    def _generate_payouts(self, results):
        """Generate payout information"""
        sorted_results = sorted(results, key=lambda x: x['finish_position'])
        
        payouts = {
            'win': {
                'horse': sorted_results[0]['horse_name'],
                'payout': round(sorted_results[0]['final_odds'] * 2, 2)
            },
            'place': {
                'horses': [sorted_results[0]['horse_name'], sorted_results[1]['horse_name']],
                'payout': round((sorted_results[0]['final_odds'] + sorted_results[1]['final_odds']) / 4 * 2, 2)
            },
            'show': {
                'horses': [sorted_results[0]['horse_name'], sorted_results[1]['horse_name'], sorted_results[2]['horse_name']],
                'payout': round((sorted_results[0]['final_odds'] + sorted_results[1]['final_odds'] + sorted_results[2]['final_odds']) / 9 * 2, 2)
            }
        }
        
        # Generate exotic payouts
        if len(results) >= 3:
            payouts['exacta'] = {
                'combination': f"{sorted_results[0]['program_number']}-{sorted_results[1]['program_number']}",
                'payout': round(np.random.uniform(20, 150), 2)
            }
            
            if len(results) >= 4:
                payouts['trifecta'] = {
                    'combination': f"{sorted_results[0]['program_number']}-{sorted_results[1]['program_number']}-{sorted_results[2]['program_number']}",
                    'payout': round(np.random.uniform(50, 500), 2)
                }
        
        return payouts

if __name__ == "__main__":
    collector = ResultsCollector()
    results = collector.generate_sample_results()
    print(f"Generated {len(results)} race results")
