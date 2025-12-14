import json
import numpy as np
from datetime import datetime, timedelta
import os

class TipstersCollector:
    def __init__(self, data_dir="data/raw/tipsters"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def generate_sample_tipster_data(self, num_tipsters=5, num_races=20):
        """Generate tipster predictions"""
        np.random.seed(42)
        
        tipster_names = [
            'Daily Racing Form', 'Brisnet', 'Equibase',
            'Timeform', 'Racing Post', 'Trackmaster'
        ]
        
        all_tipster_data = []
        
        for tipster in tipster_names[:num_tipsters]:
            tipster_predictions = {
                'tipster_name': tipster,
                'data_date': datetime.now().strftime('%Y-%m-%d'),
                'accuracy_score': round(np.random.uniform(0.15, 0.35), 3),
                'roi': round(np.random.uniform(0.85, 1.15), 2),
                'predictions': []
            }
            
            for race_id in range(1, num_races + 1):
                race_prediction = {
                    'race_id': f'R{race_id:04d}',
                    'race_date': (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d'),
                    'track': np.random.choice(['Churchill Downs', 'Santa Anita', 'Belmont Park']),
                    'selections': [],
                    'confidence_score': round(np.random.uniform(0.5, 0.9), 2),
                    'analysis': self._generate_tipster_analysis()
                }
                
                # Generate selections for this race
                num_horses = np.random.randint(6, 10)
                for rank in range(1, 4):  # Top 3 selections
                    selection = {
                        'rank': rank,
                        'horse_name': f'Horse_{race_id}_{rank}',
                        'program_number': np.random.randint(1, num_horses + 1),
                        'selection_type': ['Win', 'Place', 'Show'][rank-1],
                        'confidence': round(np.random.uniform(0.6, 0.95), 2),
                        'reason': self._generate_selection_reason(rank)
                    }
                    race_prediction['selections'].append(selection)
                
                tipster_predictions['predictions'].append(race_prediction)
            
            all_tipster_data.append(tipster_predictions)
        
        # Save to file
        filename = f"{self.data_dir}/tipsters_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(all_tipster_data, f, indent=2)
        
        print(f"Generated {len(all_tipster_data)} tipster predictions")
        return all_tipster_data
    
    def _generate_tipster_analysis(self):
        """Generate tipster race analysis"""
        analyses = [
            "Top selection has superior late speed figures and gets ideal post position.",
            "Value play on the #3 horse who is improving and gets class relief.",
            "Speed favor the rail in today's conditions. Look for early leaders to hold on.",
            "Turf specialist should handle the distance upgrade. Good value at morning line.",
            "Trainer angle strong here. First-time starter working exceptionally well.",
            "Pace scenario sets up for closers. Expect late runners to dominate."
        ]
        return np.random.choice(analyses)
    
    def _generate_selection_reason(self, rank):
        """Generate reason for selection"""
        reasons_by_rank = {
            1: ["Best speed figures", "Top trainer/jockey combo", "Class of the field"],
            2: ["Good value at odds", "Improving form", "Favorable pace setup"],
            3: ["Longshot potential", "Mud pedigree", "Fresh horse"]
        }
        return np.random.choice(reasons_by_rank.get(rank, ["Solid contender"]))
    
    def combine_tipster_consensus(self, tipster_data):
        """Combine multiple tipster predictions into consensus"""
        consensus = {}
        
        for tipster in tipster_data:
            for prediction in tipster['predictions']:
                race_id = prediction['race_id']
                
                if race_id not in consensus:
                    consensus[race_id] = {
                        'race_id': race_id,
                        'race_date': prediction['race_date'],
                        'track': prediction['track'],
                        'tipster_votes': {},
                        'consensus_picks': []
                    }
                
                # Count votes for each horse
                for selection in prediction['selections']:
                    horse_name = selection['horse_name']
                    if horse_name not in consensus[race_id]['tipster_votes']:
                        consensus[race_id]['tipster_votes'][horse_name] = 0
                    consensus[race_id]['tipster_votes'][horse_name] += 1
        
        # Create consensus picks
        for race_id, data in consensus.items():
            sorted_horses = sorted(data['tipster_votes'].items(), 
                                  key=lambda x: x[1], reverse=True)
            
            for i, (horse_name, votes) in enumerate(sorted_horses[:3]):
                data['consensus_picks'].append({
                    'rank': i + 1,
                    'horse_name': horse_name,
                    'votes': votes,
                    'consensus_strength': votes / len(tipster_data)
                })
        
        return list(consensus.values())

if __name__ == "__main__":
    collector = TipstersCollector()
    tipster_data = collector.generate_sample_tipster_data()
    consensus = collector.combine_tipster_consensus(tipster_data)
    print(f"Generated {len(tipster_data)} tipster datasets")
    print(f"Created consensus for {len(consensus)} races")
