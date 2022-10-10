from pydantic import BaseModel


class DualBladeBufftime(BaseModel):
    ultimate_dark_sight: int
    ready_to_die: int
    soul_contract: int
    restraint_ring: int
    weaponpuff_ring: int
    epic_adventure: int
    maple_world_goddess_blessing: int
    spyder_in_mirror: int
    final_cut: int
    hidden_blade: int

    def to_list(self):
        return [
            self.ultimate_dark_sight,
            self.ready_to_die,
            self.soul_contract,
            self.restraint_ring,
            self.weaponpuff_ring,
            self.epic_adventure,
            self.maple_world_goddess_blessing,
            self.spyder_in_mirror,
            self.final_cut,
            self.hidden_blade,
        ]

    def step(self):
        self.ultimate_dark_sight = max(0, self.ultimate_dark_sight - 1)
        self.ready_to_die = max(0, self.ready_to_die - 1)
        self.soul_contract = max(0, self.soul_contract - 1)
        self.restraint_ring = max(0, self.restraint_ring - 1)
        self.weaponpuff_ring = max(0, self.weaponpuff_ring - 1)
        self.epic_adventure = max(0, self.epic_adventure - 1)
        self.maple_world_goddess_blessing = max(
            0, self.maple_world_goddess_blessing - 1
        )
        self.spyder_in_mirror = max(0, self.spyder_in_mirror - 1)
        self.final_cut = max(0, self.final_cut - 1)
        self.hidden_blade = max(0, self.hidden_blade - 1)


class DualBladeCooltime(BaseModel):
    ultimate_dark_sight: int
    ready_to_die: int
    soul_contract: int
    restraint_ring: int
    weaponpuff_ring: int
    epic_adventure: int
    maple_world_goddess_blessing: int
    spyder_in_mirror: int
    final_cut: int
    hidden_blade: int
    blade_storm: int
    karma_fury: int
    blade_tornado: int
    asura: int

    def to_list(self):
        return [
            self.ultimate_dark_sight,
            self.ready_to_die,
            self.soul_contract,
            self.restraint_ring,
            self.weaponpuff_ring,
            self.epic_adventure,
            self.maple_world_goddess_blessing,
            self.spyder_in_mirror,
            self.final_cut,
            self.hidden_blade,
            self.blade_storm,
            self.karma_fury,
            self.blade_tornado,
            self.asura
        ]

    def step(self):
        self.ultimate_dark_sight = max(0, self.ultimate_dark_sight - 1)
        self.ready_to_die = max(0, self.ready_to_die - 1)
        self.soul_contract = max(0, self.soul_contract - 1)
        self.restraint_ring = max(0, self.restraint_ring - 1)
        self.weaponpuff_ring = max(0, self.weaponpuff_ring - 1)
        self.epic_adventure = max(0, self.epic_adventure - 1)
        self.maple_world_goddess_blessing = max(
            0, self.maple_world_goddess_blessing - 1
        )
        self.spyder_in_mirror = max(0, self.spyder_in_mirror - 1)
        self.final_cut = max(0, self.final_cut - 1)
        self.hidden_blade = max(0, self.hidden_blade - 1)
        self.blade_storm = max(0, self.blade_storm - 1)
        self.karma_fury = max(0, self.karma_fury - 1)
        self.blade_tornado = max(0, self.blade_tornado - 1)
        self.asura = max(0, self.asura - 1)
