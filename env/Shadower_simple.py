import gym
from gym import spaces, utils
import numpy as np

class shadowerEnvSimple(gym.Env):
    
    def __init__(self, FRAME, main_stat, sub_stat, damage, boss_damage, critical_damage, att_p, defense_ignore, final_damage, boss_defense, dealing_time):
        super(shadowerEnvSimple, self).__init__()
        
        self.penalty = -0.10
        # for reset         
        self._main_stat = main_stat
        self._sub_stat = sub_stat
        self._critical_damage = critical_damage
        self._boss_damage = boss_damage
        self._damage = damage
        self._att_p = att_p
        self._defense_ignore = defense_ignore
        self._final_damage = final_damage
        
        self.FRAME = FRAME
        self.main_stat = main_stat
        self.sub_stat = sub_stat
        self.critical_damage = critical_damage
        self.damage = damage
        self.boss_damage = boss_damage
        self.att_p = att_p
        self.defense_ignore = defense_ignore
        self.final_damage = final_damage
        self.boss_defense = boss_defense
        
        ## for learning
        self.current_time = 0
        
        self.dealing_time = dealing_time*FRAME # 2분 * 프레임 수 
        self.delay = 0 
            
        ##stat
        
        self.buff_indure_time = 20
        
        ## skill order
        ## buff or time-consuming att
        self.ultimate_dark_sight_activation_time = 0
        self.soul_contract_activation_time = 0
        self.restraint_ring_activation_time = 0
        self.weaponpuff_ring_activation_time = 0
        self.vail_of_shadow_activation_time = 0        
        self.smoke_shell_activation_time = 0        
        self.epic_adventure_activation_time = 0  
        self.vail_of_shadow_activation_time = 0   
        self.maple_world_goddess_blessing_activation_time = 0
        self.spyder_in_mirror_activation_time = 0
        self.ready_to_die_activation_time = 0
        
        ## buff or time-consuming att
        self.ready_to_die_cool_time = 0
        self.sonic_blow_cool_time = 0
        self.ultimate_dark_sight_cool_time = 0
        self.soul_contract_cool_time = 0
        self.restraint_ring_cool_time = 0
        self.weaponpuff_ring_cool_time = 0
        self.vail_of_shadow_cool_time = 0        
        self.smoke_shell_cool_time = 0        
        self.epic_adventure_cool_time = 0     
        self.maple_world_goddess_blessing_cool_time = 0
        self.spyder_in_mirror_cool_time = 0
        self.slash_shadow_formation_cool_time = 0
        self.incision_cool_time = 0
        
        
        
        self.get_state()
        
        self.num_of_state = len(self.state)
        
        self.action_name = [ 'sonic_blow', 'slash_shadow_formation', 'incision',
                            'vail_of_shadow','smoke_shell', 'epic_adventure', 'null']
       
        self.observation_space = spaces.Box(low = 0, high = 4, shape = (1,self.num_of_state))
        self.action_space = spaces.Discrete(len(self.action_name))
        
        self.state_labels = ['main_stat', 'critical_damage','boss_total_damage',
                             'att_p', 'final_damage','sonic_blow_cool','slash_shadow_cool', 'incision_cool',
                             'vail_cool', 'smoke_cool', 'epic_cool', 'uldark_cool']
    
        
    def get_state(self):
        
        ability = [self.main_stat/25000, self.critical_damage/60,
                   (self.damage + self.boss_damage )/250, self.att_p/100,
                   self.final_damage/50]
        
        cool_time = [self.sonic_blow_cool_time/(22.5*self.FRAME), 
                     self.slash_shadow_formation_cool_time/(45*self.FRAME),
                     self.incision_cool_time/(10*self.FRAME),
                     self.vail_of_shadow_cool_time/(30*self.FRAME),
                     self.smoke_shell_cool_time/(75*self.FRAME),
                     self.epic_adventure_cool_time/(60*self.FRAME),
                     self.ultimate_dark_sight_cool_time/(90*self.FRAME)]
        
        self.state = ability+ cool_time
    def reset(self):
        self.current_time = 0
        self.main_stat = self._main_stat
        self.sub_stat = self._sub_stat
        self.critical_damage = self._critical_damage
        self.boss_damage = self._boss_damage
        self.damage = self._damage
        self.att_p = self._att_p
        self.defense_ignore = self._defense_ignore
        self.final_damage = self._final_damage
        
        ## buff or time-consuming att
        self.ultimate_dark_sight_activation_time = 0
        self.soul_contract_activation_time = 0
        self.restraint_ring_activation_time = 0
        self.weaponpuff_ring_activation_time = 0
        self.vail_of_shadow_activation_time = 0        
        self.smoke_shell_activation_time = 0        
        self.epic_adventure_activation_time = 0     
        self.maple_world_goddess_blessing_activation_time = 0
        self.spyder_in_mirror_activation_time = 0
        self.dark_flare_activation_time = 0
        self.ready_to_die_activation_time = 0
        
        ## buff or time-consuming att
        self.ready_to_die_cool_time = 0
        self.sonic_blow_cool_time = 0
        self.ultimate_dark_sight_cool_time = 0
        self.soul_contract_cool_time = 0
        self.restraint_ring_cool_time = 0
        self.weaponpuff_ring_cool_time = 0
        self.vail_of_shadow_cool_time = 0        
        self.smoke_shell_cool_time = 0        
        self.epic_adventure_cool_time = 0     
        self.maple_world_goddess_blessing_cool_time = 0
        self.spyder_in_mirror_cool_time = 0
        self.dark_flare_cool_time = 0
        self.slash_shadow_formation_cool_time = 0
        self.incision_cool_time = 0
        
                      
        self.get_state()
        
        return self.state
                      
        
    def step(self, action):
                      
        self.current_time += 1
        self.delay -= 1
        self.delay = max(0 , self.delay)
        
        self.step_reward = 0
        
        if action == 0:
            self.sonic_blow()
        elif action == 1:
            self.slash_shadow_formation()
        elif action == 2:
            self.incision()
        elif action == 3:
            self.vail_of_shadow() # 버프 ?
        elif action == 4:
            self.smoke_shell()
        elif action == 5:
            self.epic_adventure() # 버프 
        elif action ==6:
            self.null_action()
            
                   
        if self.current_time % (182*self.FRAME) == 1:
            self.restraint_ring()
            self.maple_world_goddess_blessing()
            self.ultimate_dark_sight() 
        
        if self.current_time % (182*self.FRAME) == (91*self.FRAME):
            self.weaponpuff_ring()
        
        if self.current_time % (91*self.FRAME) == 1:
            self.ready_to_die()
            self.soul_contract()
        
        if self.vail_of_shadow_activation_time>0:
            vail_of_shadow_att_frame = [int(self.FRAME*(12- (i+1)* 0.85))  for i in range(14)]
            
            if self.vail_of_shadow_activation_time in vail_of_shadow_att_frame:
                ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 800,
                                'critical_damage': self.critical_damage, 'damage': self.damage, 'att_p': self.att_p,
                                'boss_damage' : self.boss_damage, 'defense_ignore': self.defense_ignore_calculator([self.defense_ignore, 20]),
                                'final_damage': self.final_damage,
                                'core_final_damage': 120, 'boss_defense' : self.boss_defense,'critical' : True}
                line_damage = self.skill_damage_calculator(ability_dict)
                self.step_reward += line_damage + 0.7* line_damage 
                
        if self.weaponpuff_ring_activation_time == 1:
            self.main_stat =  self.main_stat -14480 
        if self.restraint_ring_activation_time == 1:
            self.att_p =  self.att_p -100
        if self.soul_contract_activation_time == 1:
            self.damage =  self.damage -45
        if self.epic_adventure_activation_time == 1:
            self.damage =  self.damage -10
            
        
        if self.maple_world_goddess_blessing_activation_time == 1:
            self.main_stat =  self.main_stat -4265
            self.damage =  self.damage -20
            
             
        if self.ready_to_die_activation_time == 1:
            self.final_damage = self.final_damage_applier(self.final_damage, 36, activation = False)
            
        
        
        if self.ultimate_dark_sight_activation_time == 1:
            self.final_damage = self.final_damage_applier(self.final_damage, 31, activation = False)
            if (self.vail_of_shadow_activation_time>1 ) or (self.smoke_shell_activation_time>1 ):
                self.final_damage = self.final_damage_applier(self.final_damage, 15, activation = True)
        
        if self.vail_of_shadow_activation_time == 1 :
            if (self.ultimate_dark_sight_activation_time >1) or (self.smoke_shell_activation_time>1) :
                pass
            else:
                self.final_damage = self.final_damage_applier(self.final_damage, 15, activation = False)
        
        if self.smoke_shell_activation_time == 1 :
            if (self.vail_of_shadow_activation_time>1) or (self.ultimate_dark_sight_activation_time >1) :
                pass
            else:
                self.final_damage = self.final_damage_applier(self.final_damage, 15, activation = False)
                
                
        if self.smoke_shell_activation_time == 1:
            self.critical_damage -= 20
            
        self.ready_to_die_activation_time = max(0, self.ready_to_die_activation_time -1)
        self.ultimate_dark_sight_activation_time = max(0, self.ultimate_dark_sight_activation_time -1)
        self.maple_world_goddess_blessing_activation_time = max(0, self.maple_world_goddess_blessing_activation_time -1)
        self.spyder_in_mirror_activation_time = max(0, self.spyder_in_mirror_activation_time -1)
        
        self.soul_contract_activation_time = max(0, self.soul_contract_activation_time -1)
        self.restraint_ring_activation_time = max(0, self.restraint_ring_activation_time -1)
        self.weaponpuff_ring_activation_time = max(0, self.weaponpuff_ring_activation_time -1)
        self.vail_of_shadow_activation_time = max(0, self.vail_of_shadow_activation_time -1)        
        self.smoke_shell_activation_time = max(0, self.smoke_shell_activation_time -1)        
        self.epic_adventure_activation_time = max(0, self.epic_adventure_activation_time -1)    
        
        
        self.sonic_blow_cool_time = max(self.sonic_blow_cool_time-1, 0) 
        self.slash_shadow_formation_cool_time = max(self.slash_shadow_formation_cool_time-1, 0)
        self.incision_cool_time = max(self.incision_cool_time-1, 0)
        self.ultimate_dark_sight_cool_time = max(self.ultimate_dark_sight_cool_time-1, 0)
        self.ready_to_die_cool_time = max(self.ready_to_die_cool_time -1, 0)
        self.maple_world_goddess_blessing_cool_time = max(self.maple_world_goddess_blessing_cool_time-1, 0)
        self.spyder_in_mirror_cool_time = max(self.spyder_in_mirror_cool_time-1, 0)
        
        self.soul_contract_cool_time = max(self.soul_contract_cool_time-1, 0)
        self.restraint_ring_cool_time = max(self.restraint_ring_cool_time-1, 0)
        self.weaponpuff_ring_cool_time = max(self.weaponpuff_ring_cool_time-1, 0)
        self.vail_of_shadow_cool_time = max(self.vail_of_shadow_cool_time-1, 0)        
        self.smoke_shell_cool_time = max(self.smoke_shell_cool_time-1, 0)        
        self.epic_adventure_cool_time = max(self.epic_adventure_cool_time-1, 0)
        
            
         
        self.get_state()
                      
                      
        ep_done = self.current_time >= self.dealing_time
                      
        return (self.state, self.step_reward, ep_done, {})
        
    def incision(self): ###암살하고 쓰면 암살 선딜 후에 시작됨 
        if self.incision_cool_time > 0:
            self.step_reward += self.penalty 
            return None
        self.delay = self.att_skill_delay(0.93*self.FRAME)
        num_of_att = 35
        
        ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 1350,
                        'critical_damage': self.critical_damage, 'damage': self.damage, 'att_p': self.att_p,
                        'boss_damage' : self.boss_damage , 'defense_ignore': 100,
                        'final_damage': self.final_damage,
                        'core_final_damage': 0, 'boss_defense' : self.boss_defense,'critical' : True}
        line_damage = self.skill_damage_calculator(ability_dict)
        
        self.step_reward += line_damage*num_of_att + line_damage*num_of_att*0.7 
        
        self.incision_cool_time = self.cool_time_modifier(20*self.FRAME, 5)
        self.current_skill = 'incision'
        
    def sonic_blow(self):
        if self.sonic_blow_cool_time > 0:
            self.step_reward += self.penalty  
            return None
        self.current_skill = 'sonic_blow'
        ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 1100,
                      'critical_damage': self.critical_damage, 'damage': self.damage, 'att_p': self.att_p,
                      'boss_damage' : self.boss_damage, 'defense_ignore': 100, 'final_damage': self.final_damage,
                      'core_final_damage': 0, 'boss_defense' : self.boss_defense,'critical' : True}
        line_damage = self.skill_damage_calculator(ability_dict)
        self.step_reward += 7*15* line_damage + 7*15*0.7* line_damage
        self.sonic_blow_cool_time = self.cool_time_modifier(45*self.FRAME, 5)
    
    def slash_shadow_formation(self): 
        if self.slash_shadow_formation_cool_time > 0:
            self.step_reward += self.penalty  
            return None

        ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 935,
                    'critical_damage': self.critical_damage, 'damage': self.damage, 'att_p': self.att_p,
                    'boss_damage' : self.boss_damage, 'defense_ignore': 100, 'final_damage': self.final_damage,
                    'core_final_damage': 0, 'boss_defense' : self.boss_defense,'critical' : True}
        line_damage = self.skill_damage_calculator(ability_dict)
        self.step_reward += 8*12* line_damage
        ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 1375,
                    'critical_damage': self.critical_damage, 'damage': self.damage, 'att_p': self.att_p,
                    'boss_damage' : self.boss_damage, 'defense_ignore': 100, 'final_damage': self.final_damage,
                    'core_final_damage': 0, 'boss_defense' : self.boss_defense,'critical' : True}
        line_damage = self.skill_damage_calculator(ability_dict)
        self.step_reward += 60* line_damage

        self.current_skill = 'slash_shadow_formation'
        self.slash_shadow_formation_cool_time = self.cool_time_modifier(90*self.FRAME, 5)
        return None
        
    def ready_to_die(self): #쿨타임/ 지속시간내 발동 추가 
        # if (self.delay > 0) or (self.ready_to_die_cool_time >0):
        #     # ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 1100,
        #     #             'critical_damage': self.critical_damage, 'damage': self.damage, 'att_p': self.att_p,
        #     #             'boss_damage' : self.boss_damage, 'defense_ignore': 100, 'final_damage': self.final_damage,
        #     #             'core_final_damage': 0, 'boss_defense' : self.boss_defense,'critical' : True}
        #     # line_damage = skill_damage_calculator(ability_dict)
        #     # self.step_reward -= 1130.5/(45*self.FRAME)* line_damage # 소닉블로우에 대한 암암메의 프레임당 기대데미지  
        #     return None 
        
        self.ready_to_die_activation_time = 15 * self.FRAME
        self.final_damage = self.final_damage_applier(self.final_damage, 36)
        self.ready_to_die_cool_time = self.cool_time_modifier(75*self.FRAME, 5)
        self.current_skill = 'ready_to_die'
        return None
            
    
    def soul_contract(self):
        # if self.soul_contract_cool_time > 0 or self.delay > 0:
        #     # ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 1100,
        #     #             'critical_damage': self.critical_damage, 'damage': self.damage, 'att_p': self.att_p,
        #     #             'boss_damage' : self.boss_damage, 'defense_ignore': 100, 'final_damage': self.final_damage,
        #     #             'core_final_damage': 0, 'boss_defense' : self.boss_defense,'critical' : True}
        #     # line_damage = skill_damage_calculator(ability_dict)
        #     # self.step_reward -= 1130.5/(45*self.FRAME)* line_damage # 소닉블로우에 대한 암암메의 프레임당 기대데미지  
        #     return None
        self.current_skill = 'soul_contract'
        self.soul_contract_activation_time = self.buff_time_modifier(10, self.buff_indure_time)
        self.damage = self.damage + 45
        self.soul_contract_cool_time = self.cool_time_modifier(90*self.FRAME, 5)
        return None
    
    def maple_world_goddess_blessing(self): #수치조정 
        # if self.maple_world_goddess_blessing_cool_time > 0 or self.delay > 0:
        #     # ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 1100,
        #     #             'critical_damage': self.critical_damage, 'damage': self.damage, 'att_p': self.att_p,
        #     #             'boss_damage' : self.boss_damage, 'defense_ignore': 100, 'final_damage': self.final_damage,
        #     #             'core_final_damage': 0, 'boss_defense' : self.boss_defense,'critical' : True}
        #     # line_damage = skill_damage_calculator(ability_dict)
        #     # self.step_reward -= 1130.5/(45*self.FRAME)* line_damage # 소닉블로우에 대한 암암메의 프레임당 기대데미지  
        #     return None
        self.current_skill = 'maple_world_goddess_blessing'
        self.maple_world_goddess_blessing_activation_time = 60*self.FRAME
        self.maple_world_goddess_blessing_cool_time = self.cool_time_modifier(180*self.FRAME, 5)
        self.damage = self.damage + 20
        self.main_stat = self.main_stat + 4265
        return None
    
    def weaponpuff_ring(self): # 숨돌리기 시간추가 
        # if self.weaponpuff_ring_cool_time > 0 or self.delay > 0:
        #     # ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 1100,
        #     #             'critical_damage': self.critical_damage, 'damage': self.damage, 'att_p': self.att_p,
        #     #             'boss_damage' : self.boss_damage, 'defense_ignore': 100, 'final_damage': self.final_damage,
        #     #             'core_final_damage': 0, 'boss_defense' : self.boss_defense,'critical' : True}
        #     # line_damage = skill_damage_calculator(ability_dict)
        #     # self.step_reward -= 1130.5/(45*self.FRAME)* line_damage # 소닉블로우에 대한 암암메의 프레임당 기대데미지  
        #     return None

        self.weaponpuff_ring_activation_time = int(15*self.FRAME)
        self.current_skill = 'weaponpuff_ring'
        self.main_stat = self.main_stat + 14480
        
        self.weaponpuff_ring_cool_time = 180*self.FRAME
        return None
    
    def restraint_ring(self): # 숨돌리기 시간추가  
        # if self.restraint_ring_cool_time > 0 or self.delay > 0:
        #     # ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 1100,
        #     #             'critical_damage': self.critical_damage, 'damage': self.damage, 'att_p': self.att_p,
        #     #             'boss_damage' : self.boss_damage, 'defense_ignore': 100, 'final_damage': self.final_damage,
        #     #             'core_final_damage': 0, 'boss_defense' : self.boss_defense,'critical' : True}
        #     # line_damage = skill_damage_calculator(ability_dict)
        #     # self.step_reward -= 1130.5/(45*self.FRAME)* line_damage # 소닉블로우에 대한 암암메의 프레임당 기대데미지  
        #     return None
        self.restraint_ring_activation_time = int(15*self.FRAME)
        self.current_skill = 'restraint_ring'
        self.att_p = self.att_p + 100
        self.restraint_ring_cool_time = 180*self.FRAME
        return None
    
    def epic_adventure(self): # 숨돌리기 시간추가  
        if self.epic_adventure_cool_time > 0:
            self.step_reward += self.penalty  
            return None
        self.current_skill = 'restraint_ring'
        self.epic_adventure_activation_time = int(60*self.FRAME)
        self.damage = self.damage + 10
        self.epic_adventure_cool_time = 120*self.FRAME
        return None
    
    
    def ultimate_dark_sight(self): #0.9 초 준비 1.6초동안 순수 7타 1.6
        # if self.ultimate_dark_sight_cool_time > 0:
        #     self.step_reward += self.penalty 
        #     return None
        self.current_skill = 'ultimate_dark_sight'
        self.ultimate_dark_sight_activation_time = 30*self.FRAME
        self.ultimate_dark_sight_cool_time = self.cool_time_modifier(190*self.FRAME, 5)  
        
        
        if (self.vail_of_shadow_activation_time>0 ) or (self.smoke_shell_activation_time>0 ):
            self.final_damage = self.final_damage_applier(self.final_damage, 15, activation = False)
            self.final_damage = self.final_damage_applier(self.final_damage, 31, activation = True)
        else:
            self.final_damage = self.final_damage_applier(self.final_damage, 31, activation = True)
        
        return None

    def smoke_shell(self):
        if self.smoke_shell_cool_time > 0:
            self.step_reward += self.penalty  
            return None
        self.current_skill = 'smoke_shell'
        self.smoke_shell_activation_time = int(30.1*self.FRAME)
        self.smoke_shell_cool_time = self.cool_time_modifier(150*self.FRAME, 5)
        self.critical_damage = self.critical_damage+ 20 
        if (self.vail_of_shadow_activation_time>0 ) or (self.ultimate_dark_sight_activation_time>0 ):
            return None
        else:
            self.final_damage = self.final_damage_applier(self.final_damage, 15, activation = True)
        return None
    
    def vail_of_shadow(self):
        if self.vail_of_shadow_cool_time > 0:
            self.step_reward += self.penalty 
            return None
        self.current_skill = 'vail_of_shadow'
        self.vail_of_shadow_activation_time = int(12.1*self.FRAME)
        self.vail_of_shadow_cool_time = 60*self.FRAME
        if (self.smoke_shell_activation_time>0 ) or (self.ultimate_dark_sight_activation_time>0 ):
            return None
        else:
            self.final_damage = self.final_damage_applier(self.final_damage, 15, activation = True)
        return None

    
    def spyder_in_mirror(self):
        if self.spyder_in_mirror_cool_time > 0:
            # ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 1100,
            #             'critical_damage': self.critical_damage, 'damage': self.damage, 'att_p': self.att_p,
            #             'boss_damage' : self.boss_damage, 'defense_ignore': 100, 'final_damage': self.final_damage,
            #             'core_final_damage': 0, 'boss_defense' : self.boss_defense,'critical' : True}
            # line_damage = skill_damage_calculator(ability_dict)
            # self.step_reward -= 1130.5/(45*self.FRAME)* line_damage # 소닉블로우에 대한 암암메의 프레임당 기대데미지  
            return None
            
        self.current_skill = 'spyder_in_mirror'
        self.spyder_in_mirror_activation_time = int(50*self.FRAME)
        self.spyder_in_mirror_cool_time = self.cool_time_modifier(250*self.FRAME, 5)
        self.delay = int(0.96*self.FRAME)
        
        num_of_att= 12
        ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 990,
                        'critical_damage': self.critical_damage, 'damage': self.damage, 'att_p': self.att_p,
                        'boss_damage' : self.boss_damage , 'defense_ignore': self.defense_ignore,
                        'final_damage': self.final_damage,
                        'core_final_damage': 0, 'boss_defense' : self.boss_defense,'critical' : True}
        line_damage = self.skill_damage_calculator(ability_dict)
        
        self.step_reward += line_damage*num_of_att
        return None
        
    def null_action(self):
        ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 270,
                        'critical_damage': self.critical_damage, 'damage': self.damage+20, 'att_p': self.att_p,
                        'boss_damage' : self.boss_damage+20, 'defense_ignore': self.defense_ignore_calculator([self.defense_ignore, 28]),
                        'final_damage': self.final_damage,
                        'core_final_damage': 120, 'boss_defense' : self.boss_defense,'critical' : True}
        line_damage = 6 * 1.7*self.skill_damage_calculator(ability_dict)

        ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 490,
                        'critical_damage': self.critical_damage, 'damage': self.damage+20, 'att_p': self.att_p,
                        'boss_damage' : self.boss_damage+20, 'defense_ignore': self.defense_ignore_calculator([self.defense_ignore, 28]),
                        'final_damage': self. final_damage_applier(self.final_damage,50),
                        'core_final_damage': 120, 'boss_defense' : self.boss_defense,'critical' : True}
        line_damage2 = 6* 1.7*self.skill_damage_calculator(ability_dict)

        ability_dict = {'main_stat': self.main_stat, 'sub_stat' : self.sub_stat, 'skill_damage': 100,
                        'critical_damage': self.critical_damage, 'damage': self.damage+20, 'att_p': self.att_p,
                        'boss_damage' : self.boss_damage+ 30, 'defense_ignore': self.defense_ignore_calculator([self.defense_ignore, 20]),
                        'final_damage': self.final_damage,
                        'core_final_damage': 180, 'boss_defense' : self.boss_defense,'critical' : True}
        line_damage3 = 9.6*2* 1.7*self.skill_damage_calculator(ability_dict)

        self.step_reward +=  36*(line_damage + line_damage2 +line_damage3)/ (30*self.FRAME) *0.99
        return None
    
    
    def render(self, mode = 'human'):
        return None
    def close(self):
        return None
    
    def skill_damage_calculator(self, ability_dict):
        main_stat = ability_dict['main_stat']
        sub_stat  = ability_dict['sub_stat']
        skill_damage = ability_dict['skill_damage']
        critical_damage  = ability_dict['critical_damage']
        damage = ability_dict['damage']
        att_p = ability_dict['att_p']
        boss_damage = ability_dict['boss_damage']
        defense_ignore = ability_dict['defense_ignore']
        final_damage = ability_dict['final_damage']
        core_final_damage = ability_dict['core_final_damage']
        boss_defense = ability_dict['boss_defense']
        critical = ability_dict['critical'] 
        
        if critical:
            critical_constant = ((35+critical_damage)/100) +1
        else:
            critical_constant = 1
        defense_modification = 1- (boss_defense* (100 - defense_ignore)/100)/100
        
        skill_damage = ((main_stat*4+ sub_stat) * 2582 * 1.3 * 1 /100)*(skill_damage/100)* critical_constant* ((100+att_p)/100) *\
        ((100+damage+ boss_damage)/100) *defense_modification*((100+final_damage)/100) *((100+core_final_damage)/100)
        return skill_damage/(10**11)

    def att_skill_delay(self, delay):
        return int((delay*3/4) //30 *30+ int((delay*3/4)%30 != 0)*30)

    def final_damage_applier(self, pre_final_damage, add_final_damage, activation = True):
        final_damage_val = 100+ pre_final_damage
        if activation:
            return (final_damage_val*(100+add_final_damage)/100 - 100)
        else:
            return (final_damage_val *100/(add_final_damage+100) - 100)
        
    def buff_time_modifier(self, buff_time, time_amount):
        return int(buff_time*(100+time_amount)/100)
        

    def cool_time_modifier(self, cool_time, time_amount):
        return int(cool_time*(100-time_amount)/100)

    def defense_ignore_calculator(self, defense_ignore_list):
        defense = 100 
        for di in defense_ignore_list:
            defense = defense * (100-di)/100
        return 100- defense
    
    
                