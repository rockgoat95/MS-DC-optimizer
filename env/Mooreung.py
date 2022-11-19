import gym 


class Mooreung(gym.Env):
    def __init__(
        self,
        job_env,
        monsters,
        remain_time = 12*60 +30,
        HP_divider = 1e8
    ):
        
        self.job_env = job_env
        self.remain_time = remain_time
        self.monster_dict = {}
        
        for i, row in monsters.iterrows():
            self.monster_dict[row['step']] = [row['level'], row['HP'], row['boss_defense'], row['deal_rate']] 
        
        self.HP_divider = HP_divider
        
        self.now_step = 41
        self.step_info = self.monster_dict[self.now_step]
    
    def step(self, action):
        state, reward, _  = self.job_env.step(action)
    
    
        self.step_info[2] -= reward
        
        if self.step_info[2] <= 0 :
            self.now_step +=1
            self.step_info = self.monster_dict[self.now_step]
            
    def _get_state(self,job_state):
        
        return job_state + []
        
        