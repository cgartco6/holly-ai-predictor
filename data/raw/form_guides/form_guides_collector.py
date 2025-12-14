import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class FormGuidesCollector:
    def __init__(self, data_dir="data/raw/form_guides"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def generate_sample_form_guides(self, num_horses=100):
        """Generate comprehensive form guides"""
        np.random.seed(42)
        
        all_form_guides = []
        
        for horse_id in range(1, num_horses + 1):
            form_guide = {
                'horse_id': f'H{horse_id:04d}',
                'horse_name': f'Form_Horse_{horse_id}',
                'update_date': datetime.now().strftime('%Y-%m-%d'),
                'basic_info': self._generate_basic_info(horse_id),
                'pedigree': self._generate_pedigree(),
                'race_record': self._generate_race_record(horse_id),
                'workout_history': self._generate_workout_history(),
                'veterinary_records': self._generate_vet_records(),
                'handicapper_notes': self._generate_handicapper_notes(),
                'speed_figures': self._generate_speed_figures(),
                'pace_analysis': self._generate_pace_analysis(),
                'track_biases': self._generate_track_biases(),
                'trainer_patterns': self._generate_trainer_patterns(),
                'jockey_stats': self._generate_jockey_stats()
            }
            
            all_form_guides.append(form_guide)
        
        # Save to file
        filename = f"{self.data_dir}/form_guides_{datetime.now().strftime('%Y%m')}.json"
        with open(filename, 'w') as f:
            json.dump(all_form_guides, f, indent=2)
        
        print(f"Generated {len(all_form_guides)} form guides")
        return all_form_guides
    
    def _generate_basic_info(self, horse_id):
        """Generate basic horse information"""
        return {
            'age': np.random.randint(2, 8),
            'sex': np.random.choice(['Colt', 'Filly', 'Gelding', 'Mare']),
            'color': np.random.choice(['Bay', 'Chestnut', 'Gray', 'Black']),
            'foal_date': (datetime.now() - timedelta(days=np.random.randint(730, 2920))).strftime('%Y-%m-%d'),
            'breeder': np.random.choice(['Kentucky Breeders', 'California Stud', 'Florida Farm']),
            'current_trainer': np.random.choice(['Bob Baffert', 'Todd Pletcher', 'Chad Brown', 'Steve Asmussen']),
            'current_owner': np.random.choice(['Stable A', 'Farm B', 'Racing Syndicate C']),
            'career_earnings': np.random.randint(10000, 500000),
            'starts_wins_places': {
                'starts': np.random.randint(5, 30),
                'wins': np.random.randint(1, 8),
                'places': np.random.randint(2, 10),
                'shows': np.random.randint(3, 12)
            }
        }
    
    def _generate_pedigree(self):
        """Generate pedigree information"""
        sires = ['Into Mischief', 'Tapit', 'American Pharoah', 'Curlin', 'Medaglia d\'Oro']
        dams = ['Dream Girl', 'Star Princess', 'Moon Queen', 'Sky Dancer']
        
        return {
            'sire': np.random.choice(sires),
            'dam': np.random.choice(dams),
            'sire_sire': np.random.choice(['Speightstown', 'Storm Cat', 'A.P. Indy']),
            'dam_sire': np.random.choice(['Danzig', 'Seattle Slew', 'Mr. Prospector']),
            'breeding_rating': round(np.random.uniform(70, 95), 1),
            'distance_preference': np.random.choice(['Sprinter', 'Miler', 'Router']),
            'surface_preference': np.random.choice(['Dirt', 'Turf', 'All-weather']),
            'mud_pedigree': round(np.random.uniform(0.5, 1.0), 2)
        }
    
    def _generate_race_record(self, horse_id):
        """Generate detailed race record"""
        races = []
        num_races = np.random.randint(5, 20)
        
        for i in range(num_races):
            race_date = datetime.now() - timedelta(days=np.random.randint(30, 365 * 2))
            
            race = {
                'race_date': race_date.strftime('%Y-%m-%d'),
                'track': np.random.choice(['CD', 'SA', 'BEL', 'GP', 'KEE']),
                'distance': np.random.choice([5.0, 6.0, 7.0, 8.0, 9.0]),
                'surface': np.random.choice(['Dirt', 'Turf']),
                'class': np.random.choice(['Mdn', 'Clm', 'Alw', 'Stk', 'G1', 'G2', 'G3']),
                'finish': np.random.randint(1, 12),
                'beaten': round(np.random.uniform(0, 10), 1),
                'field_size': np.random.randint(6, 12),
                'odds': round(np.random.uniform(1.5, 25.0), 2),
                'weight': np.random.randint(1140, 1220),
                'jockey': np.random.choice(['J. Velazquez', 'M. Smith', 'J. Castellano']),
                'comment': np.random.choice([
                    'Won driving', 'Rallied late', 'Evenly', 'No factor',
                    'Closed well', 'Faded stretch', 'Bumped start'
                ]),
                'pace_position': {
                    'start': np.random.randint(1, 8),
                    'quarter': np.random.randint(1, 8),
                    'half': np.random.randint(1, 8),
                    'stretch': np.random.randint(1, 8)
                }
            }
            races.append(race)
        
        return races
    
    def _generate_workout_history(self):
        """Generate workout/training history"""
        workouts = []
        num_workouts = np.random.randint(10, 30)
        
        for i in range(num_workouts):
            workout_date = datetime.now() - timedelta(days=np.random.randint(1, 60))
            
            workout = {
                'date': workout_date.strftime('%Y-%m-%d'),
                'track': np.random.choice(['CD', 'SA', 'BEL']),
                'distance': np.random.choice([3, 4, 5, 6]),
                'time': f'{np.random.randint(35, 38)}.{np.random.randint(0, 60):02d}',
                'surface': np.random.choice(['Dirt', 'Turf']),
                'condition': np.random.choice(['Fast', 'Good', 'Sloppy']),
                'rank': np.random.randint(1, 20),
                'total_workers': np.random.randint(10, 30),
                'comment': np.random.choice(['Breezing', 'Handily', 'Bullet work', 'Maintenance'])
            }
            workouts.append(workout)
        
        return workouts
    
    def _generate_vet_records(self):
        """Generate veterinary records"""
        return {
            'last_vet_check': (datetime.now() - timedelta(days=np.random.randint(1, 30))).strftime('%Y-%m-%d'),
            'soundness_rating': round(np.random.uniform(0.7, 1.0), 2),
            'recent_issues': np.random.choice(['None', 'Minor foot soreness', 'Cough', 'Clean bill']),
            'medications': np.random.choice(['Lasix', 'Bute', 'None']),
            'layoff_reason': np.random.choice(['None', 'Rest', 'Minor injury', 'Freshening']),
            'days_since_last_race': np.random.randint(14, 90)
        }
    
    def _generate_handicapper_notes(self):
        """Generate handicapper notes"""
        notes = [
            "Improving with each start. Should handle class rise.",
            "Needs lead or press pace. Doesn't rate well.",
            "Excellent mud pedigree. Watch for off tracks.",
            "Best going two turns. Might be short today.",
            "Trainer excels with layoffs. Ready to fire.",
            "Jockey switch positive. Gets along with horse."
        ]
        return {
            'strengths': np.random.choice(notes, size=2, replace=False).tolist(),
            'weaknesses': np.random.choice(notes, size=2, replace=False).tolist(),
            'overall_assessment': np.random.choice(notes),
            'projected_improvement': round(np.random.uniform(0.8, 1.2), 2)
        }
    
    def _generate_speed_figures(self):
        """Generate speed figure analysis"""
        return {
            'beyer_average': np.random.randint(75, 90),
            'beyer_last_3': [np.random.randint(70, 95) for _ in range(3)],
            'beyer_best': np.random.randint(85, 100),
            'beyer_trend': round(np.random.uniform(-5, 5), 1),
            'speed_variant': round(np.random.uniform(0.9, 1.1), 2),
            'distance_specific': {
                'sprint': np.random.randint(70, 85),
                'route': np.random.randint(75, 90)
            }
        }
    
    def _generate_pace_analysis(self):
        """Generate pace analysis"""
        return {
            'running_style': np.random.choice(['Early', 'Presser', 'Closer', 'Stalker']),
            'early_speed_points': np.random.randint(70, 95),
            'late_speed_points': np.random.randint(70, 95),
            'pace_figure_average': np.random.randint(75, 90),
            'preferred_scenario': np.random.choice(['Fast pace', 'Slow pace', 'No preference']),
            'final_fraction_rating': round(np.random.uniform(0.8, 1.2), 2)
        }
    
    def _generate_track_biases(self):
        """Generate track-specific performance"""
        tracks = ['CD', 'SA', 'BEL', 'GP', 'KEE']
        
        biases = {}
        for track in tracks:
            if np.random.random() > 0.3:
                biases[track] = {
                    'starts': np.random.randint(1, 5),
                    'wins': np.random.randint(0, 3),
                    'average_finish': round(np.random.uniform(2.5, 6.5), 1),
                    'preference': round(np.random.uniform(0.8, 1.2), 2)
                }
        
        return biases
    
    def _generate_trainer_patterns(self):
        """Generate trainer-specific patterns"""
        patterns = [
            'Strong with layoffs',
            'Excellent first time starter',
            'Good with route to sprint',
            'Positive blinkers on',
            'Strong off claim',
            'Good with turf to dirt'
        ]
        
        return {
            'trainer_name': np.random.choice(['Bob Baffert', 'Todd Pletcher', 'Chad Brown']),
            'win_percentage': round(np.random.uniform(0.15, 0.35), 3),
            'roi': round(np.random.uniform(0.85, 1.15), 2),
            'key_patterns': np.random.choice(patterns, size=3, replace=False).tolist(),
            'recent_form': round(np.random.uniform(0.8, 1.2), 2)
        }
    
    def _generate_jockey_stats(self):
        """Generate jockey statistics"""
        return {
            'jockey_name': np.random.choice(['John Velazquez', 'Mike Smith', 'Javier Castellano']),
            'win_percentage': round(np.random.uniform(0.15, 0.25), 3),
            'mount_earnings': np.random.randint(1000000, 10000000),
            'track_specific': {
                'CD': round(np.random.uniform(0.15, 0.25), 3),
                'SA': round(np.random.uniform(0.15, 0.25), 3)
            },
            'trainer_combo': round(np.random.uniform(0.2, 0.4), 3)
        }

if __name__ == "__main__":
    collector = FormGuidesCollector()
    form_guides = collector.generate_sample_form_guides(50)
    print(f"Generated {len(form_guides)} comprehensive form guides")
