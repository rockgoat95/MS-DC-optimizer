def skill_damage_calculator(ability_dict):
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