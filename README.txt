Lauren Kaffa
16-8-2021
Assignment AE4350

Code consists of 3 RL algorithms: PG, QL and DQN (DQN is not included in report due to page restrictions)

PG:
	- PG_Agent_Class.py			Class of the PG Agent
	- PG_Agent_Training.py			training of PG Agent class object
	- PG_Agent_Trained_model.py		shows result of trained model. Displays average, min., max. and variance of played games
	- PG_sensitivity_plots.py		Generates the sensitivity analysis plots for the PG models, the final model training performance and the policy entropy	during training

QL:
	- Q_leaning.py				Generates and updates Q-table
	- Q_get_discrete_state_function.py	Defines the custom state interval boundaries and return discretized state given a "continuous" state
	- Q_learning_display.py			shows result of trained model. Displays average, min., max. and variance of played games				
	- Q_sensitivity_plots.py		Generates the sensitivity analysis plots for the PG models and the final model training performance

DQN:
	- DQN_Agent_class.py			Class of the DQN Agent	
	- DQN_Agent_Training.py			training of DQN Agent class object	
	- DQN_display.py			

Behaviour analysis:
	- Behavioural_analysis.py		Plot all actions of the QL- and PG method from Behavioural_data
						Data is plotted for: horizontal velocity vs vertical velocity
								     angle vs. angular rate

Environment:
	- Environment.py			Environment only, agent just takes random actions. Used to obatin score baseline.

		
