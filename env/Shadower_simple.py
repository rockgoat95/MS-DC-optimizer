import gym
from gym import spaces, utils
import numpy as np

from module.utils import *
from module.schema.Shadower import *

from copy import deepcopy


class ShadowerEnvSimple(gym.Env):
    def __init__(
        self,
        FRAME,
        ability,
        boss_defense,
        dealing_time,
        common_attack_rate=0.9,
        reward_divider=1e9,
        test=False,
    ):
        super(ShadowerEnvSimple, self).__init__()

        self.test = test
        self.penalty = -0.10
        self.reward_divider = reward_divider

        self.ability = ability
        # for reset
        self._ability = deepcopy(ability)
        self.common_attack_rate = common_attack_rate
        self.weapon_constant = 1.3
        self.boss_defense = boss_defense

        # for learning
        self.FRAME = FRAME

        ## for learning
        self.current_time = 0

        self.dealing_time = dealing_time * FRAME  # 딜 타임 * 프레임 수

        self.reset(return_state=False)

        self.num_of_state = len(self.state)

        self.observation_space = spaces.Box(low=0, high=1, shape=( self.num_of_state,))

        self.action_name = [
            "sonic_blow",
            "slash_shadow_formation",
            "incision",
            "vail_of_shadow",
            "smoke_shell",
            "epic_adventure",
            "spyder_in_mirror",
            "null",
        ]

        self.action_space = spaces.Discrete(len(self.action_name))

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
            "sinmi_cool",
        ]

        self.vail_of_shadow_att_frame = [
            int(self.FRAME * (12 - (i + 1) * 0.85)) for i in range(14)
        ]

        self.spyder_in_mirror_att_frame = []
        for k in range(50, 9, -8):
            self.spyder_in_mirror_att_frame += [k - 3 - (i + 1) for i in range(5)]
        self.spyder_in_mirror_att_frame = [
            frame * self.FRAME for frame in self.spyder_in_mirror_att_frame
        ]

    def update_state(self):

        ## apply approximate min max scaling
        ability_state = [
            (self.ability.main_stat - self._ability.main_stat)
            / (self._ability.weapon_puff_inc + self._ability.maple_goddess2_inc),
            (self.ability.critical_damage - self._ability.critical_damage) / 20,
            (
                self.ability.damage
                + self.ability.boss_damage
                - self._ability.damage
                - self._ability.boss_damage
            )
            / 75,
            (self.ability.att_p - self._ability.att_p) / 100,
            (self.ability.final_damage - self._ability.final_damage)
            / (
                (100 + self._ability.final_damage) * 1.31 * 1.36
                - 100
                - self._ability.final_damage
            ),
        ]

        cool_time_state = [
            self.cool_time.sonic_blow / cool_time_modifier(45 * self.FRAME, 5),
            self.cool_time.slash_shadow_formation
            / cool_time_modifier(90 * self.FRAME, 5),
            self.cool_time.incision / cool_time_modifier(20 * self.FRAME, 5),
            self.cool_time.vail_of_shadow / cool_time_modifier(60 * self.FRAME, 5),
            self.cool_time.smoke_shell / cool_time_modifier(150 * self.FRAME, 5),
            self.cool_time.epic_adventure / cool_time_modifier(120 * self.FRAME, 5),
            self.cool_time.ultimate_dark_sight
            / cool_time_modifier(190 * self.FRAME, 5),
            self.cool_time.spyder_in_mirror / cool_time_modifier(250 * self.FRAME, 5),
        ]

        self.state = np.array(
            [self.current_time / (self.dealing_time * self.FRAME)]
            + ability_state
            + cool_time_state
        )

    def reset(self, return_state=True):
        self.current_time = 0

        self.ability = deepcopy(self._ability)

        ## buff or time-consuming att
        self.buff_time = ShadowerBufftime(
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

        self.cool_time = ShadowerCooltime(
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

        self.update_state()
        if return_state:
            return self.state

    def step(self, action):

        self.current_time += 1

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
            self.spyder_in_mirror()
        elif action == 7:
            self.common_attack()  # 평타

        # 사전 정보를 이용해 필수적인 타이밍에 버프 실행
        self.auto_buff()

        # 버프시간 체크 및 시간 종료시 버프효과 제거
        self.check_buff_and_deact()

        self.time_lag_attack()

        # cool, buff time flowing
        self.cool_time.step()
        self.buff_time.step()

        self.update_state()

        ep_done = self.current_time >= self.dealing_time

        if not self.test:
            if self.step_reward >= 0:
                self.step_reward /= self.reward_divider
            else:
                pass

            return (self.state, self.step_reward, ep_done, {})
        else:
            return (self.state, max(0, self.step_reward), ep_done, {})

    def incision(self):
        if self.cool_time.incision > 0:
            self.step_reward += self.penalty
            return None
        num_of_att = 35

        ability_in_skill = deepcopy(self.ability)
        ability_in_skill.add(defense_ignore=100)
        line_damage = skill_damage_calculator(
            ability_in_skill,
            skill_damage=1350,
            boss_defense=self.boss_defense,
            weapon_constant=self.weapon_constant,
        )
        self.step_reward += line_damage * num_of_att + line_damage * num_of_att * 0.7

        self.cool_time.incision = cool_time_modifier(20 * self.FRAME, 5)

    def sonic_blow(self):
        if self.cool_time.sonic_blow > 0:
            self.step_reward += self.penalty
            return None
        num_of_att = 7 * 15

        ability_in_skill = deepcopy(self.ability)
        ability_in_skill.add(defense_ignore=100)
        line_damage = skill_damage_calculator(
            ability_in_skill,
            skill_damage=1100,
            boss_defense=self.boss_defense,
            weapon_constant=self.weapon_constant,
        )

        self.step_reward += num_of_att * line_damage + num_of_att * 0.7 * line_damage
        self.cool_time.sonic_blow = cool_time_modifier(45 * self.FRAME, 5)

    def slash_shadow_formation(self):
        if self.cool_time.slash_shadow_formation > 0:
            self.step_reward += self.penalty
            return None
        ability_in_skill = deepcopy(self.ability)
        ability_in_skill.add(defense_ignore=100)
        line_damage = skill_damage_calculator(
            ability_in_skill,
            skill_damage=935,
            boss_defense=self.boss_defense,
            weapon_constant=self.weapon_constant,
        )
        self.step_reward += 8 * 12 * line_damage

        ability_in_skill = deepcopy(self.ability)
        ability_in_skill.add(defense_ignore=100)
        line_damage = skill_damage_calculator(
            ability_in_skill,
            skill_damage=1375,
            boss_defense=self.boss_defense,
            weapon_constant=self.weapon_constant,
        )
        self.step_reward += 60 * line_damage

        self.cool_time.slash_shadow_formation = cool_time_modifier(90 * self.FRAME, 5)
        return None

    def ready_to_die(self):
        self.buff_time.ready_to_die = 15 * self.FRAME
        self.ability.add(final_damage=36)
        self.cool_time.ready_to_die = cool_time_modifier(75 * self.FRAME, 5)
        return None

    def soul_contract(self):
        self.buff_time.soul_contract = buff_time_modifier(
            10 * self.FRAME, self.ability.buff_indure_time
        )
        self.ability.add(damage=45)
        self.cool_time.soul_contract = cool_time_modifier(90 * self.FRAME, 5)
        return None

    def maple_world_goddess_blessing(self):
        self.buff_time.maple_world_goddess_blessing = 60 * self.FRAME
        self.ability.add(main_stat=self.ability.maple_goddess2_inc, damage=20)
        self.cool_time.maple_world_goddess_blessing = cool_time_modifier(
            180 * self.FRAME, 5
        )
        return None

    def weaponpuff_ring(self):
        self.buff_time.weaponpuff_ring = 15 * self.FRAME
        self.ability.add(main_stat=self.ability.weapon_puff_inc)
        self.cool_time.weaponpuff_ring = 180 * self.FRAME
        return None

    def restraint_ring(self):
        self.buff_time.restraint_ring = 15 * self.FRAME
        self.ability.add(att_p=100)
        self.cool_time.restraint_ring = 180 * self.FRAME
        return None

    def epic_adventure(self):

        if self.cool_time.epic_adventure > 0:
            self.step_reward += self.penalty
            return None

        self.buff_time.epic_adventure = 60 * self.FRAME
        self.ability.add(damage=10)
        self.cool_time.epic_adventure = cool_time_modifier(120 * self.FRAME, 5)
        return None

    def ultimate_dark_sight(self):
        self.buff_time.ultimate_dark_sight = 30 * self.FRAME
        self.cool_time.ultimate_dark_sight = cool_time_modifier(190 * self.FRAME, 5)

        if (self.buff_time.vail_of_shadow > 0) or (self.buff_time.smoke_shell > 0):
            self.ability.add(final_damage=-15)
            self.ability.add(final_damage=31)
        else:
            self.ability.add(final_damage=31)

        return None

    def smoke_shell(self):
        if self.cool_time.smoke_shell > 0:
            self.step_reward += self.penalty
            return None

        self.buff_time.smoke_shell = 30 * self.FRAME
        self.ability.add(critical_damage=20)
        self.cool_time.smoke_shell = cool_time_modifier(150 * self.FRAME, 5)

        if (self.buff_time.vail_of_shadow > 0) or (
            self.buff_time.ultimate_dark_sight > 0
        ):
            return None
        else:
            self.ability.add(final_damage=15)
        return None

    def vail_of_shadow(self):
        if self.cool_time.vail_of_shadow > 0:
            self.step_reward += self.penalty
            return None
        self.buff_time.vail_of_shadow = 12 * self.FRAME
        self.cool_time.vail_of_shadow = cool_time_modifier(60 * self.FRAME, 5)
        if (self.buff_time.smoke_shell > 0) or (self.buff_time.ultimate_dark_sight > 0):
            pass
        else:
            self.ability.add(final_damage=15)
        return None

    def spyder_in_mirror(self):
        if self.cool_time.spyder_in_mirror > 0:
            return None

        num_of_att = 12

        self.buff_time.spyder_in_mirror = int(50 * self.FRAME)
        self.cool_time.spyder_in_mirror = cool_time_modifier(250 * self.FRAME, 5)

        line_damage = skill_damage_calculator(
            self.ability, 990, self.boss_defense, weapon_constant=self.weapon_constant
        )

        self.step_reward += line_damage * num_of_att
        return None

    def common_attack(self):
        ability_in_skill = deepcopy(self.ability)
        ability_in_skill.add(defense_ignore=28, damage=20, boss_damage=20)
        line_damage = skill_damage_calculator(
            ability_in_skill,
            skill_damage=270,
            boss_defense=self.boss_defense,
            weapon_constant=self.weapon_constant,
            core_final_damage=120,
        )
        assacination1_damage = 6 * 1.7 * line_damage

        ability_in_skill = deepcopy(self.ability)
        ability_in_skill.add(
            defense_ignore=28, damage=20, boss_damage=20, final_damage=50
        )
        line_damage = skill_damage_calculator(
            ability_in_skill,
            skill_damage=490,
            boss_defense=self.boss_defense,
            weapon_constant=self.weapon_constant,
            core_final_damage=120,
        )
        assacination2_damage = 6 * 1.7 * line_damage

        ability_in_skill = deepcopy(self.ability)
        ability_in_skill.add(defense_ignore=20, damage=20, boss_damage=30)
        line_damage = skill_damage_calculator(
            ability_in_skill,
            skill_damage=100,
            boss_defense=self.boss_defense,
            weapon_constant=self.weapon_constant,
            core_final_damage=180,
        )
        meso_explosion_damage = 9.6 * 2 * 1.7 * line_damage

        # 암암메 최대 분당 타수 36을 참고하여 한번에 계산
        self.step_reward += (
            36
            * (assacination1_damage + assacination2_damage + meso_explosion_damage)
            / (60 * self.FRAME)
            * self.common_attack_rate
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
            if self.buff_time.vail_of_shadow in self.vail_of_shadow_att_frame:
                ability_in_skill = deepcopy(self.ability)
                ability_in_skill.add(defense_ignore=20)
                line_damage = skill_damage_calculator(
                    ability_in_skill,
                    skill_damage=800,
                    boss_defense=self.boss_defense,
                    core_final_damage=120,
                    weapon_constant=self.weapon_constant,
                )
                self.step_reward += line_damage + 0.7 * line_damage

        if self.buff_time.spyder_in_mirror > 0:
            if self.buff_time.spyder_in_mirror in self.spyder_in_mirror_att_frame:
                line_damage = skill_damage_calculator(
                    self.ability,
                    skill_damage=385,
                    boss_defense=self.boss_defense,
                    weapon_constant=self.weapon_constant,
                )
                self.step_reward += 8 * line_damage

    def check_buff_and_deact(self):
        # buff deact
        if self.buff_time.weaponpuff_ring == 1:
            self.ability.add(main_stat=-self.ability.weapon_puff_inc)
        if self.buff_time.restraint_ring == 1:
            self.ability.add(att_p=-100)
        if self.buff_time.soul_contract == 1:
            self.ability.add(damage=-45)
        if self.buff_time.epic_adventure == 1:
            self.ability.add(damage=-10)

        if self.buff_time.maple_world_goddess_blessing == 1:
            self.ability.add(main_stat=-self.ability.maple_goddess2_inc)
            self.ability.add(damage=-20)
        if self.buff_time.ready_to_die == 1:
            self.ability.add(final_damage=-36)
        if self.buff_time.ultimate_dark_sight == 1:
            self.ability.add(final_damage=-31)
            if (self.buff_time.vail_of_shadow > 1) or (self.buff_time.smoke_shell > 1):
                self.ability.add(final_damage=15)

        if self.buff_time.vail_of_shadow == 1:
            if (self.buff_time.ultimate_dark_sight > 1) or (
                self.buff_time.smoke_shell > 1
            ):
                pass
            else:
                self.ability.add(final_damage=-15)

        if self.buff_time.smoke_shell == 1:
            if (self.buff_time.vail_of_shadow > 1) or (
                self.buff_time.ultimate_dark_sight > 1
            ):
                pass
            else:
                self.ability.add(final_damage=-15)

        if self.buff_time.smoke_shell == 1:
            self.ability.add(critical_damage=-20)

    def render(self, mode="human"):
        return None

    def close(self):
        return None
