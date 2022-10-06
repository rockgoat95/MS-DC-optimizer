import gym
from gym import spaces, utils
import numpy as np

from module.utils import *
from module.schema.shadower import *

from copy import deepcopy


class shadowerEnvSimple(gym.Env):
    def __init__(self, FRAME, ability, boss_defense, dealing_time):
        super(shadowerEnvSimple, self).__init__()

        self.penalty = -0.10

        self.ability = ability
        # for reset
        self._ability = deepcopy(ability)

        # for learning
        self.FRAME = FRAME

        ## for learning
        self.current_time = 0

        self.dealing_time = dealing_time * FRAME  # 딜 타임 * 프레임 수

        self.get_state()

        self.num_of_state = len(self.state)

        self.observation_space = spaces.Box(low=0, high=4, shape=(1, self.num_of_state))
        self.action_space = spaces.Discrete(len(self.action_name))

        self.action_name = [
            "sonic_blow",
            "slash_shadow_formation",
            "incision",
            "vail_of_shadow",
            "smoke_shell",
            "epic_adventure",
            "null",
        ]

        self.state_labels = [
            "main_stat",
            "critical_damage",
            "boss_total_damage",
            "att_p",
            "final_damage",
            "sonic_blow_cool",
            "slash_shadow_cool",
            "incision_cool",
            "vail_cool",
            "smoke_cool",
            "epic_cool",
            "uldark_cool",
        ]

    def get_state(self):

        ## apply approximate min max scaling
        ability = [
            self.main_stat / 25000,
            self.critical_damage / 60,
            (self.damage + self.boss_damage) / 250,
            self.att_p / 100,
            self.final_damage / 50,
        ]

        cool_time = [
            self.sonic_blow_cool_time / (45 * self.FRAME),
            self.slash_shadow_formation_cool_time / (90 * self.FRAME),
            self.incision_cool_time / (20 * self.FRAME),
            self.vail_of_shadow_cool_time / (60 * self.FRAME),
            self.smoke_shell_cool_time / (150 * self.FRAME),
            self.epic_adventure_cool_time / (120 * self.FRAME),
            self.ultimate_dark_sight_cool_time / (190 * self.FRAME),
        ]

        self.state = ability + cool_time

    def reset(self):
        self.current_time = 0
        self.ability = deepcopy(self._ability)

        ## buff or time-consuming att
        self.buff_time = Shadower_Bufftime(
            ultimate_dark_sight=0,
            ready_to_die=0,
            soul_contract=0,
            restraint_ring=0,
            weaponpuff_ring=0,
            vail_of_shadow=0,
            smoke_shell=0,
            epic_adventure=0,
            maple_world_goddess_blessing=0,
            spyder_in_mirror=0,
            dark_flare=0,
        )

        self.cool_time = Shadower_Cooltime(
            ultimate_dark_sight=0,
            ready_to_die=0,
            soul_contract=0,
            restraint_ring=0,
            weaponpuff_ring=0,
            vail_of_shadow=0,
            smoke_shell=0,
            epic_adventure=0,
            maple_world_goddess_blessing=0,
            spyder_in_mirror=0,
            dark_flare=0,
            sonic_blow=0,
            slash_shadow_formation=0,
            incision=0,
        )

        self.get_state()

        return self.state

    def step(self, action):

        self.current_time += 1
        self.delay = max(0, self.delay - 1)

        self.step_reward = 0

        if action == 0:
            self.sonic_blow()
        elif action == 1:
            self.slash_shadow_formation()
        elif action == 2:
            self.incision()
        elif action == 3:
            self.vail_of_shadow()
        elif action == 4:
            self.smoke_shell()
        elif action == 5:
            self.epic_adventure()
        elif action == 6:
            self.null_action()  # 평타

        self.auto_buff()
        self.deact_buff()
       
        self.time_lag_attack()
        
        # cool, buff time flowing 
        self.cool_time.step()
        self.buff_time.step()

        self.get_state()
        ep_done = self.current_time >= self.dealing_time

        return (self.state, self.step_reward, ep_done, {self.step_reward})

    def incision(self):
        if self.cool_time.incision > 0:
            self.step_reward += self.penalty
            return None
        num_of_att = 35
        
        ability_in_skill = deepcopy(self.ablility)
        ability_in_skill.add(defense_ignore = 100)
        line_damage = skill_damage_calculator(
            ability_in_skill,
            skill_damage=1350,
            boss_defense=300,
            weapon_constant=1.3,
        )
        self.step_reward += line_damage * num_of_att + line_damage * num_of_att * 0.7

        self.incision_cool_time = cool_time_modifier(20 * self.FRAME, 5)

    def sonic_blow(self):
        if self.sonic_blow_cool_time > 0:
            self.step_reward += self.penalty
            return None
        self.current_skill = "sonic_blow"
        ability_dict = {
            "main_stat": self.main_stat,
            "sub_stat": self.sub_stat,
            "skill_damage": 1100,
            "critical_damage": self.critical_damage,
            "damage": self.damage,
            "att_p": self.att_p,
            "boss_damage": self.boss_damage,
            "defense_ignore": 100,
            "final_damage": self.final_damage,
            "core_final_damage": 0,
            "boss_defense": self.boss_defense,
            "critical": True,
        }
        line_damage = skill_damage_calculator(ability_dict)
        self.step_reward += 7 * 15 * line_damage + 7 * 15 * 0.7 * line_damage
        self.sonic_blow_cool_time = cool_time_modifier(45 * self.FRAME, 5)

    def slash_shadow_formation(self):
        if self.slash_shadow_formation_cool_time > 0:
            self.step_reward += self.penalty
            return None

        ability_dict = {
            "main_stat": self.main_stat,
            "sub_stat": self.sub_stat,
            "skill_damage": 935,
            "critical_damage": self.critical_damage,
            "damage": self.damage,
            "att_p": self.att_p,
            "boss_damage": self.boss_damage,
            "defense_ignore": 100,
            "final_damage": self.final_damage,
            "core_final_damage": 0,
            "boss_defense": self.boss_defense,
            "critical": True,
        }
        line_damage = skill_damage_calculator(ability_dict)
        self.step_reward += 8 * 12 * line_damage
        ability_dict = {
            "main_stat": self.main_stat,
            "sub_stat": self.sub_stat,
            "skill_damage": 1375,
            "critical_damage": self.critical_damage,
            "damage": self.damage,
            "att_p": self.att_p,
            "boss_damage": self.boss_damage,
            "defense_ignore": 100,
            "final_damage": self.final_damage,
            "core_final_damage": 0,
            "boss_defense": self.boss_defense,
            "critical": True,
        }
        line_damage = skill_damage_calculator(ability_dict)
        self.step_reward += 60 * line_damage

        self.slash_shadow_formation_cool_time = cool_time_modifier(90 * self.FRAME, 5)
        return None

    def ready_to_die(self):
        self.buff_time.ready_to_die = 15 * self.FRAME
        self.ability.add(final_damage = 36)
        self.cool_time.ready_to_die  = cool_time_modifier(75 * self.FRAME, 5)
        return None

    def soul_contract(self):
        self.buff_time.soul_contract = 15 * self.FRAME
        self.ability.add(damage = 45)
        self.cool_time.soul_contract  = cool_time_modifier(90 * self.FRAME, 5)
        return None

    def maple_world_goddess_blessing(self):
        self.current_skill = "maple_world_goddess_blessing"
        self.maple_world_goddess_blessing_activation_time = 60 * self.FRAME
        self.maple_world_goddess_blessing_cool_time = cool_time_modifier(
            180 * self.FRAME, 5
        )
        self.damage = self.damage + 20
        self.main_stat = self.main_stat + 4265
        return None

    def weaponpuff_ring(self):
        self.weaponpuff_ring_activation_time = int(15 * self.FRAME)
        self.current_skill = "weaponpuff_ring"
        self.main_stat = self.main_stat + 14480

        self.weaponpuff_ring_cool_time = 180 * self.FRAME
        return None

    def restraint_ring(self):
        self.restraint_ring_activation_time = int(15 * self.FRAME)
        self.current_skill = "restraint_ring"
        self.att_p = self.att_p + 100
        self.restraint_ring_cool_time = 180 * self.FRAME
        return None

    def epic_adventure(self):
        if self.epic_adventure_cool_time > 0:
            self.step_reward += self.penalty
            return None
        self.current_skill = "restraint_ring"
        self.epic_adventure_activation_time = int(60 * self.FRAME)
        self.damage = self.damage + 10
        self.epic_adventure_cool_time = 120 * self.FRAME
        return None

    def ultimate_dark_sight(self):
        self.current_skill = "ultimate_dark_sight"
        self.ultimate_dark_sight_activation_time = 30 * self.FRAME
        self.ultimate_dark_sight_cool_time = cool_time_modifier(190 * self.FRAME, 5)

        if (self.vail_of_shadow_activation_time > 0) or (
            self.smoke_shell_activation_time > 0
        ):
            self.final_damage = final_damage_applier(
                self.final_damage, 15, activation=False
            )
            self.final_damage = final_damage_applier(
                self.final_damage, 31, activation=True
            )
        else:
            self.final_damage = final_damage_applier(
                self.final_damage, 31, activation=True
            )

        return None

    def smoke_shell(self):
        if self.smoke_shell_cool_time > 0:
            self.step_reward += self.penalty
            return None
        self.current_skill = "smoke_shell"
        self.smoke_shell_activation_time = int(30.1 * self.FRAME)
        self.smoke_shell_cool_time = cool_time_modifier(150 * self.FRAME, 5)
        self.critical_damage = self.critical_damage + 20
        if (self.vail_of_shadow_activation_time > 0) or (
            self.ultimate_dark_sight_activation_time > 0
        ):
            return None
        else:
            self.final_damage = final_damage_applier(
                self.final_damage, 15, activation=True
            )
        return None

    def vail_of_shadow(self):
        if self.vail_of_shadow_cool_time > 0:
            self.step_reward += self.penalty
            return None
        self.current_skill = "vail_of_shadow"
        self.vail_of_shadow_activation_time = int(12.1 * self.FRAME)
        self.vail_of_shadow_cool_time = 60 * self.FRAME
        if (self.smoke_shell_activation_time > 0) or (
            self.ultimate_dark_sight_activation_time > 0
        ):
            return None
        else:
            self.final_damage = final_damage_applier(
                self.final_damage, 15, activation=True
            )
        return None

    def spyder_in_mirror(self):
        if self.spyder_in_mirror_cool_time > 0:
            return None

        self.current_skill = "spyder_in_mirror"
        self.spyder_in_mirror_activation_time = int(50 * self.FRAME)
        self.spyder_in_mirror_cool_time = cool_time_modifier(250 * self.FRAME, 5)
        self.delay = int(0.96 * self.FRAME)

        num_of_att = 12
        ability_dict = {
            "main_stat": self.main_stat,
            "sub_stat": self.sub_stat,
            "skill_damage": 990,
            "critical_damage": self.critical_damage,
            "damage": self.damage,
            "att_p": self.att_p,
            "boss_damage": self.boss_damage,
            "defense_ignore": self.defense_ignore,
            "final_damage": self.final_damage,
            "core_final_damage": 0,
            "boss_defense": self.boss_defense,
            "critical": True,
        }
        line_damage = skill_damage_calculator(ability_dict)

        self.step_reward += line_damage * num_of_att
        return None

    def null_action(self):
        ability_dict = {
            "main_stat": self.main_stat,
            "sub_stat": self.sub_stat,
            "skill_damage": 270,
            "critical_damage": self.critical_damage,
            "damage": self.damage + 20,
            "att_p": self.att_p,
            "boss_damage": self.boss_damage + 20,
            "defense_ignore": defense_ignore_calculator([self.defense_ignore, 28]),
            "final_damage": self.final_damage,
            "core_final_damage": 120,
            "boss_defense": self.boss_defense,
            "critical": True,
        }
        line_damage = 6 * 1.7 * skill_damage_calculator(ability_dict)

        ability_dict = {
            "main_stat": self.main_stat,
            "sub_stat": self.sub_stat,
            "skill_damage": 490,
            "critical_damage": self.critical_damage,
            "damage": self.damage + 20,
            "att_p": self.att_p,
            "boss_damage": self.boss_damage + 20,
            "defense_ignore": defense_ignore_calculator([self.defense_ignore, 28]),
            "final_damage": final_damage_applier(self.final_damage, 50),
            "core_final_damage": 120,
            "boss_defense": self.boss_defense,
            "critical": True,
        }
        line_damage2 = 6 * 1.7 * skill_damage_calculator(ability_dict)

        ability_dict = {
            "main_stat": self.main_stat,
            "sub_stat": self.sub_stat,
            "skill_damage": 100,
            "critical_damage": self.critical_damage,
            "damage": self.damage + 20,
            "att_p": self.att_p,
            "boss_damage": self.boss_damage + 30,
            "defense_ignore": defense_ignore_calculator([self.defense_ignore, 20]),
            "final_damage": self.final_damage,
            "core_final_damage": 180,
            "boss_defense": self.boss_defense,
            "critical": True,
        }
        line_damage3 = 9.6 * 2 * 1.7 * skill_damage_calculator(ability_dict)

        # 암암메 최대 분당 타수 36을 참고하여 한번에 계산
        self.step_reward += (
            36 * (line_damage + line_damage2 + line_damage3) / (60 * self.FRAME) * 0.90
        )
        return None

    def auto_buff(self):
        # auto buff
        if self.current_time % (182 * self.FRAME) == 1:
            self.restraint_ring()
            self.maple_world_goddess_blessing()
            self.ultimate_dark_sight()

        if self.current_time % (182 * self.FRAME) == (91 * self.FRAME):
            self.weaponpuff_ring()

        if self.current_time % (91 * self.FRAME) == 1:
            self.ready_to_die()
            self.soul_contract()

    def time_lag_attack(self):
        # darksight logic
        if self.buff_time.vail_of_shadow > 0:
            vail_of_shadow_att_frame = [
                int(self.FRAME * (12 - (i + 1) * 0.85)) for i in range(14)
            ]

            if self.buff_time.vail_of_shadow in vail_of_shadow_att_frame:
                ability_in_skill = deepcopy(self.ability)
                ability_in_skill.add(defense_ignore = 20)
                line_damage = skill_damage_calculator(
                    ability_in_skill,
                    skill_damage=800,
                    boss_defense=300,
                    weapon_constant=1.3,
                )
                self.step_reward += line_damage + 0.7 * line_damage

    def deact_buff(self):
        # buff deact
        if self.buff_time.weaponpuff_ring == 1:
            self.ability.main_stat = (
                self.ability.main_stat - self.ability.weapon_puff_inc
            )
        if self.buff_time.restraint_ring == 1:
            self.ability.add(att_p = -100)
        if self.buff_time.soul_contract == 1:
            self.ability.add(damage = -45)
        if self.buff_time.epic_adventure == 1:
            self.ability.add(damage = -10)

        if self.buff_time.maple_world_goddess_blessing == 1:
            self.ability.add(main_stat = - self.ability.maple_goddess2_inc)
            self.ability.add(damage = -20)
        if self.buff_time.ready_to_die == 1:
            self.ability.add(final_damage = -36)
        if self.buff_time.ultimate_dark_sight == 1:
            self.ability.final_damage = final_damage_applier(
                self.ability.final_damage, 31, activation=False
            )
            if (self.buff_time.vail_of_shadow > 1) or (self.buff_time.smoke_shell > 1):
                self.ability.final_damage = final_damage_applier(
                    self.ability.final_damage, 15, activation=True
                )

        if self.buff_time.vail_of_shadow == 1:
            if (self.buff_time.ultimate_dark_sight > 1) or (
                self.buff_time.smoke_shell > 1
            ):
                pass
            else:
                self.ability.final_damage = final_damage_applier(
                    self.ability.final_damage, 15, activation=False
                )

        if self.buff_time.smoke_shell == 1:
            if (self.buff_time.vail_of_shadow > 1) or (
                self.buff_time.ultimate_dark_sight > 1
            ):
                pass
            else:
                self.ability.final_damage = final_damage_applier(
                    self.ability.final_damage, 15, activation=False
                )

        if self.buff_time.smoke_shell == 1:
            self.ability.critical_damage -= 20

    def render(self, mode="human"):
        return None

    def close(self):
        return None
