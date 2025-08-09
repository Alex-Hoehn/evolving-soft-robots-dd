import os
import json
import time
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt

class SEDDExperimentLogger:
    def __init__(self, exp_name, exp_dir):
        self.exp_name = exp_name
        self.log_dir = os.path.join(exp_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_data = {
            'exp_name': exp_name,
            'start_time': datetime.datetime.now().isoformat(),
            'end_time': None,
            'generations': [],
            'task_model_losses': [],
            'overall_stats': {}
        }

        log_file = os.path.join(self.log_dir, 'experiment.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            force=True
        )

        self.logger = logging.getLogger(f"SEDD_{exp_name}")
        self.logger.info(f"Starting SEDD experiment: {exp_name}")

    def log_generation_start(self, gen_num):
        self.gen_start_time = time.time()
        self.logger.info(f" ============== GENERATION {gen_num} ============== ")

    def log_generation_complete(self, gen_num, scores, improved=False, robot_scores=None):
        if not scores:
            self.logger.error(f"Generation {gen_num} had no scores to log.")
            return

        gen_time_secs = time.time() - self.gen_start_time

        gen_data = {
            'generation': gen_num,
            'timestamp': datetime.datetime.now().isoformat(),
            'duration_minutes': gen_time_secs / 60,
            'num_robots': len(scores),
            'used_task_model': improved,
            'performance_stats': {
                'best_score': float(max(scores)),
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
            },
            'scores': scores,
            'robot_ranking': sorted(robot_scores, key=lambda x: x[1], reverse=True) if robot_scores else []
        }
        self.log_data['generations'].append(gen_data)

        stats = gen_data['performance_stats']
        self.logger.info(f"GENERATION {gen_num} COMPLETE in {gen_time_secs/60:.1f} mins.")
        self.logger.info(f"   Scores -> Best: {stats['best_score']:.4f}, Mean: {stats['mean_score']:.4f}")

        # Compare with the previous generation
        if gen_num > 0:
            prev_best = self.log_data['generations'][-2]['performance_stats']['best_score']
            improvement = stats['best_score'] - prev_best
            self.logger.info(f"     Change from last gen: {improvement:+.4f}")

        # Show top performers
        if gen_data['robot_ranking']:
            self.logger.info("   Top Performers:")
            for i, (robot_id, score) in enumerate(gen_data['robot_ranking'][:3]):
                self.logger.info(f"     #{i+1}: Robot {robot_id} (Score: {score:.4f})")

        self._save_results()

    def log_task_model_update(self, loss):
        self.log_data['task_model_losses'].append(loss)
        self.logger.info(f"Task Model updated. Loss: {loss:.4f}")

    def log_experiment_complete(self):
        self.log_data['end_time'] = datetime.datetime.now().isoformat()
        start_dt = datetime.datetime.fromisoformat(self.log_data['start_time'])
        end_dt = datetime.datetime.fromisoformat(self.log_data['end_time'])

        all_gens = self.log_data['generations']
        if not all_gens:
            self.logger.warning("Experiment completed with no generations logged.")
            return

        all_scores = [s for gen in all_gens for s in gen['scores']]

        # Calculate final overall statistics
        stats = {
            'total_duration_hours': (end_dt - start_dt).total_seconds() / 3600,
            'total_generations': len(all_gens),
            'best_score_overall': float(max(all_scores)) if all_scores else 0,
            'total_improvement': (all_gens[-1]['performance_stats']['best_score'] -
                                  all_gens[0]['performance_stats']['best_score']),
            'avg_gen_duration_minutes': float(np.mean([g['duration_minutes'] for g in all_gens]))
        }
        self.log_data['overall_stats'] = stats

        # Log final summary
        self.logger.info(f"~~~~~~~~~~~~~~~~~~~~~~~\nEXPERIMENT COMPLETE - SUMMARY\n~~~~~~~~~~~~~~~~~~~~~~~")
        self.logger.info(f"Total Time: {stats['total_duration_hours']:.2f} hours")
        self.logger.info(f"Best Score Achieved: {stats['best_score_overall']:.4f}")
        self.logger.info(f"Total Improvement: {stats['total_improvement']:.4f}")

        self._save_results()

    def _save_results(self):
        json_path = os.path.join(self.log_dir, 'experiment_data.json')
        with open(json_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)
