from module.schema.Type import Ability

a = Ability(main_stat = 1, sub_stat = 1, damage = 1, boss_damage = 1, att_p = 1, defense_ignore = 1, critical_damage= 1, final_damage = 1, buff_indure_time= 1,
            maple_goddess_inc = 1, weapon_att = 1)
def skill_damage_calculator(ability : Ability, skill_damage : int, boss_defense : int , critical : bool, core_final_damage :int , weapon_constant : int = 1 , job_constant : int = 1):
    if critical:
        critical_constant = ((35+ability.critical_damage)/100) +1
    else:
        critical_constant = 1
    defense_modification = max(0 , 1- (boss_defense* (100 - ability.defense_ignore)/100)/100)
    
    attack_damage = ((ability.main_stat*4+ ability.sub_stat) * 2582 * weapon_constant * job_constant/100)*(skill_damage/100)* critical_constant* ((100+ability.att_p)/100) *\
    ((100+ability.damage+ ability.boss_damage)/100) *defense_modification*((100+ability.final_damage)/100) *((100+core_final_damage)/100)
    
    return int(attack_damage)

skill_damage_calculator(a,skill_damage = 100, boss_defense = 300, critical = True, core_final_damage = 0)

def att_skill_delay(delay):
    return int((delay*3/4) //30 *30+ int((delay*3/4)%30 != 0)*30)

def final_damage_applier(pre_final_damage, add_final_damage, activation = True):
    final_damage_val = 100+ pre_final_damage
    if activation:
        return (final_damage_val*(100+add_final_damage)/100 - 100)
    else:
        return (final_damage_val *100/(add_final_damage+100) - 100)
    
def buff_time_modifier(buff_time, buff_time_endure):
    return int(buff_time*(100+buff_time_endure)/100)
    

def cool_time_modifier(cool_time, reduce_cool_time):
    
    return int(cool_time*(100-reduce_cool_time)/100)

def defense_ignore_calculator(defense_ignore_list):
    defense = 100 
    for di in defense_ignore_list:
        defense = defense * (100-di)/100
    return 100- defense