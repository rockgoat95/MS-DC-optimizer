import gym
from gym import spaces, utils
import numpy as np

# 모듈 폴더 내 함수 참고 
from module.utils import *
from module.schema.Shadower import *

from copy import deepcopy

# 리워드, 데미지가 혼용되어 쓰였는데 이 환경에서는 거의 같은 말로 보시면 됩니다. 

class ShadowerEnvSimple(gym.Env):
    def __init__(
        self,
        FRAME : int,
        ability : Ability,
        boss_defense : int,
        dealing_time : int,
        common_attack_rate : float = 0.9,
        reward_divider : int = 1e9,
        test=False,
    ):
        '''
        FRAME : 초당 프레임 수 
        ability : 능력치 
        boss_defense : 대상 보스 방어율
        dealing_time : 딜링 시간(초) // 이후 프레임수로 변환함
        common_attack_rate : 평타 반영 비율 
        
        현재 환경에서 모든 스킬에 delay가 고려되지 않음 
        common_attack에 해당하는 액션을 선택하면 평타에 해당하는 암암메의 프레임당 데미지가 리워드로 적용됨 
        이 경우 리워드 보정이 필요한데 현재 0.75정도로 설정되어 학습됨. 즉 효과를 0.75배 정도로 고려하겠다는 의미. 
        
        쉽게 말하면 90초 중에 1초(5프레임)에 모든 극딜을 우겨 넣고 89초 동안 평타를 넣는 경우, 
        딜레이가 고려되지 않아 평타의 효과가 너무 커져버리므로 이를 보정해주기 위함 
        ==> 이 부분은 무릉 프로젝트에서 제거하고 진행할 것이지만, 어느정도 싸이클은 묶어서 고려하는 것이 학습되기 쉬운 방향으로 보임 
        
        ( 결과로 제공된 사진에서 나머지 액션은 모두 common attack임 )
        
        reward_divider : 리워드(데미지) 조정수치  
        
        '''
        super(ShadowerEnvSimple, self).__init__()

        # 학습 or 테스트 여부 (테스트시에는 실제 들어가는 데미지, 학습 시에는 조정된 데미지가 리워드가됨.) 
        self.test = test
        # 쿨타임일때 누르면 부여할 패널티 
        self.penalty = -0.15
        # 리워드 조정수치 (데미지 / reward_divider가 리워드로 적용됨)
        self.reward_divider = reward_divider

        # 입력된 능력치 저장 
        self.ability = ability
        # for reset
        
        # 초기 능력치 저장 
        self._ability = deepcopy(ability)
        
        # 평타 반영 비율 
        self.common_attack_rate = common_attack_rate
        
        # 무기상수 (직업별로 상이)
        self.weapon_constant = 1.3
        
        # 현재 때리는 보스 방무 
        self.boss_defense = boss_defense

        # 환경의 프레임 (5프레임이면 0.2초 단위로 각각의 행동을 선택함)
        self.FRAME = FRAME

        ## 현재 진행된 시간 current_time 이 아래의 self.dealing_time 과 같아질 때 에피소드가 종료됨
        self.current_time = 0

        # 전체 딜링 프레임 (dealing_time은 초 단위)
        self.dealing_time = dealing_time * FRAME  # 딜 타임 * 프레임 수

        # 환경 리셋 
        self.reset(return_state=False)

        # 학습 시 전달 상태의 차원 수 
        self.num_of_state = len(self.state)

        # 전달 되는 상태의 공간 정의 (기본 환경 구성 요소)
        self.observation_space = spaces.Box(low=0, high=1, shape=( self.num_of_state,))


        # 전달되는 상태의 라벨링 (상태가 잘 전달되는지 확인할 때 사용됨)
        self.state_labels = [
            "current_time",
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

        # 액션 번호에 따른 액션의 라벨링
        self.action_name = [
            "sonic_blow", # 소닉블로우 
            "slash_shadow_formation", # 멸귀참영진
            "incision", # 절개 
            "vail_of_shadow", 
            "smoke_shell", # 연막탄
            "epic_adventure",
            "spyder_in_mirror",
            "null", # 평타 
        ]

        # 액션 공간의 차원 수 
        self.action_space = spaces.Discrete(len(self.action_name))

        # 배오섀의 공격 주기 계산 
        # 배오섀가 적용되면 배오셰 적용시간이 입력되는데 적용시간이 주기와 맞을때마다 딜이 들어가게끔 계산된 부분 
        # ex) self.vail_of_shadow_att_frame = [2,5,8] 이면 배오섀 사용 후 2프레임 5프레임 8프레임 뒤에 데미지 들어가게끔 하기 위해서.. 
        self.vail_of_shadow_att_frame = [
            int(self.FRAME * (12 - (i + 1) * 0.85)) for i in range(14)
        ]

        # 스인미의 공격 주기 계산 
        # 배오섀와 같은 의도로 적용됨 
        self.spyder_in_mirror_att_frame = []
        for k in range(50, 9, -8):
            self.spyder_in_mirror_att_frame += [k - 3 - (i + 1) for i in range(5)]
        self.spyder_in_mirror_att_frame = [
            frame * self.FRAME for frame in self.spyder_in_mirror_att_frame
        ]

    # 에피소드 초기화 (gym.Env의 고유한 형식)
    def reset(self, return_state=True):
        # 딜타임 초기화 
        self.current_time = 0
        # 능력치 초기화 
        self.ability = deepcopy(self._ability)

        ## 스킬 지속시간 
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

        # 스킬 쿨타임 
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

        # 상태 계산 
        self.update_state()
        
        # 초기 상태 리턴 여부 체크 및 리턴 
        if return_state:
            return self.state

    def update_state(self):
        # 입력상태 업데이트 
        # min max scaling 이 적용됨 (이론상 최대치와 최저치를 이용,  가장 낮은 값이 0 높은 값이 1)
        
        # 능력치 상태 
        ability_state = [
            (self.ability.main_stat - self._ability.main_stat)
            / (self._ability.weapon_puff_inc + self._ability.maple_goddess2_inc), # 주스탯 , 부스탯은 일단 고려 안함
            (self.ability.critical_damage - self._ability.critical_damage) / 20, # 크뎀 
            (
                self.ability.damage
                + self.ability.boss_damage
                - self._ability.damage
                - self._ability.boss_damage
            )
            / 75, # 보총뎀 (어차피 합쳐서 계산되니 Agent가 이해하기 좋은 방향인 것 같음 )
            (self.ability.att_p - self._ability.att_p) / 100, # 공 퍼
            (self.ability.final_damage - self._ability.final_damage)
            / (
                (100 + self._ability.final_damage) * 1.31 * 1.36 #1.31== 얼닼사, 1.36 == 레투다로 오르는 수치 
                - 100
                - self._ability.final_damage
            ), # 최종뎀 
        ]

        # 쿨타임 상태
        # cool_time_modifier 는 메르세데스 유니온 효과를 적용하기 위해 적용됨. 
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

        # 최종 전달 상태  // 현재시간을 상태에 추가한 이유는 이 환경은 딜사이클 최적화가 목적이므로 시간에 오버피팅하는게 좋을 것 같아서 넣어본 부분 입니다.. 
        self.state = np.array(
            [self.current_time / (self.dealing_time * self.FRAME)]
            + ability_state
            + cool_time_state
        )

    # 현재 상태에서 액션이 취해졌을 때 
    # 변하는 상태, 리워드, 종료 여부 등을 전달하는 함수 (gym.Env의 고유한 형식)
    def step(self, action):
        
        # 1프레임 만큼 시간 지남 
        self.current_time += 1

        # 이 스텝에서 리워드 초기화 
        self.step_reward = 0
        # 각 액션에 따라 self.step_reward에 적용된 데미지 저장 

        # 액션 선택 
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

        # 사전 정보를 이용해 자동적으로 적절한 타이밍에 버프 실행 (ex, 180초 주기로 리레 적용)
        self.auto_buff()

        # 버프시간 체크 및 시간 종료시 버프효과 제거
        self.check_buff_and_deact()

        # 시간차 공격이 잇는 경우에 적절한 프레임에 reward적용 (섀도어의 경우 배오섀, 스인미) 
        self.time_lag_attack()

        # cool, buff time flowing
        # 쿨타임 버프타임이 1프레임 감소함 
        self.cool_time.step()
        self.buff_time.step()

        # 상태 계산 
        self.update_state()

        # 에피소드 종료 여부 
        ep_done = self.current_time >= self.dealing_time

        # 학습시에는 패널티 적용, 리워드 조정 // 테스트시에는 패널티적용 x, 리워드 조정 x 
        if not self.test:
            if self.step_reward >= 0:
                self.step_reward /= self.reward_divider
            else:
                pass

            return (self.state, self.step_reward, ep_done, {})
        else:
            return (self.state, max(0, self.step_reward), ep_done, {})

    # 절개 (원래는 조건이 있지만 크게 어려운부분 아니므로 조건 없이 발동이 가능하다고 가정)
    def incision(self):
        # 쿨타임 있을 때 사용하면 패널티 부여받고 종료 
        if self.cool_time.incision > 0:
            self.step_reward += self.penalty
            return None
        # 절개 타수 
        num_of_att = 35

        # 스킬 사용시 적용 스탯 // 절개는 방무 100이라 이와 같이 적용 
        ability_in_skill = deepcopy(self.ability)
        ability_in_skill.add(defense_ignore=100)
        
        # 줄당 데미지 계산
        line_damage = skill_damage_calculator(
            ability_in_skill,
            skill_damage=1350,
            boss_defense=self.boss_defense,
            weapon_constant=self.weapon_constant,
        )
        
        # 데미지 계산 0.7부분은 섀파 적용 딜 
        self.step_reward += line_damage * num_of_att + line_damage * num_of_att * 0.7
        
        # 쿨타임 부여 
        self.cool_time.incision = cool_time_modifier(20 * self.FRAME, 5)

    # 아래부터 유사한 부분은 주석 생략 
    
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
        
        # 공격 별 데미지가 다르게 들어가므로 두번 계산 하여 self.step_reward에 더하여 계산 
        # 섀파 적용 안됨 
        
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
        # 주기적으로 자동 적용되므로 패널티 부여 코드 생략됨 (아래에 자동적용 버프도 패널티 부여 코드 생략)
        # 무릉에 적용하면 패널티 부여적용되어야합니다! 
        # 버프적용 시간 입력 15초 
        self.buff_time.ready_to_die = 15 * self.FRAME
        # 버프에 따른 스탯 조정 
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
        # 메용 2 상승량의 경우 계산이 어려워 입력으로 받긴하지만 계산 공식이 있을 것으로 추정.. 
        self.ability.add(main_stat=self.ability.maple_goddess2_inc, damage=20)
        self.cool_time.maple_world_goddess_blessing = cool_time_modifier(
            180 * self.FRAME, 5
        )
        return None

    def weaponpuff_ring(self):
        self.buff_time.weaponpuff_ring = 15 * self.FRAME
        # 웨펖 상승량도 계산 못했지만 주스탯 퍼 + 무기공 받으면 계산 가능할 것으로 생각됨  
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

        # 얼닼사의 경우 연막탄, 배오섀로 인한 다크사이트적용과 중첩될 여지가 있으므로 아래와 같이 룰에 다라 최종뎀 상승량 적용 
        if (self.buff_time.vail_of_shadow > 0) or (self.buff_time.smoke_shell > 0):
            self.ability.add(final_damage=-15)
            self.ability.add(final_damage=31)
        else:
            self.ability.add(final_damage=31)

        return None

    # 연막탄과 배오섀의 경우 사용시 자동으로 다크사이트가 사용된다고 설정 
    # 다크사이트 사용시 최종뎀 15퍼 버프가 있음
    def smoke_shell(self):
        if self.cool_time.smoke_shell > 0:
            self.step_reward += self.penalty
            return None

        self.buff_time.smoke_shell = 30 * self.FRAME
        self.ability.add(critical_damage=20)# 크뎀 버프 있음
        self.cool_time.smoke_shell = cool_time_modifier(150 * self.FRAME, 5)

        # 얼닼사와 같은 이유 
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

        # 스인미 처음 터지는 데미지 // 다리 데미지는 time_lag_attack에서 적용 
        line_damage = skill_damage_calculator(
            self.ability, 990, self.boss_defense, weapon_constant=self.weapon_constant
        )

        self.step_reward += line_damage * num_of_att
        return None
    
    # 평타 
    def common_attack(self):
        # 암살 1타 
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

        # 암살 2타 
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

        # 메익 (평균 사출량인 9.6개 적용 )
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

        # 암암메 최대 분당 타수 36임을 참고하여 프레임당 데미지 계산 
        self.step_reward += (
            36
            * (assacination1_damage + assacination2_damage + meso_explosion_damage)
            / (60 * self.FRAME)
            * self.common_attack_rate
        )
        return None

    # 자동 버프 적용 (얼닼사 주기인 180.5 초에 맞추려고 182 초 극딜버프 91초 준극딜버프 사용되도록 설정 )
    # 무릉 적용시 해당 부분은 제거해야할 것으로 보임 
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

    # 스인미 배오섀 시간차 데미지 적용 부분 
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

    # 버프 시간을 체크하고 효과를 원상태로 돌리는 부분 
    # 즉 다음 프레임에서 버프효과가 없기 때문에 버프효과를 푸는 부분임 
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
            
        # 다크사이트 로직 .. 
        # 배오섀 또는 연막탄있을 때 다크사이트 적용되므로 15퍼 이지만 
        # 얼닼사 사용되면 다크사이트 적용부분 제거하고 최종뎀 31퍼이기 때문에 아래와 같은 로직 적용 
        # 듀블 섀도어 말고는 필요없는 로직 
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
            self.ability.add(critical_damage=-20)

    def render(self, mode="human"):
        return None

    def close(self):
        return None
