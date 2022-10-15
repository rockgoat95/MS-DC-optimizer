from pydantic import BaseModel


class ShadowerBufftime(BaseModel):
    ultimate_dark_sight: int
    ready_to_die: int
    soul_contract: int
    restraint_ring: int
    weaponpuff_ring: int
    vail_of_shadow: int
    smoke_shell: int
    epic_adventure: int
    maple_world_goddess_blessing: int
    spyder_in_mirror: int
    dark_flare: int

    def to_list(self):
        return [
            self.ultimate_dark_sight,
            self.ready_to_die,
            self.soul_contract,
            self.restraint_ring,
            self.weaponpuff_ring,
            self.vail_of_shadow,
            self.smoke_shell,
            self.epic_adventure,
            self.maple_world_goddess_blessing,
            self.spyder_in_mirror,
            self.dark_flare,
        ]

    def step(self):
        self.ultimate_dark_sight = max(0, self.ultimate_dark_sight - 1)
        self.ready_to_die = max(0, self.ready_to_die - 1)
        self.soul_contract = max(0, self.soul_contract - 1)
        self.restraint_ring = max(0, self.restraint_ring - 1)
        self.weaponpuff_ring = max(0, self.weaponpuff_ring - 1)
        self.vail_of_shadow = max(0, self.vail_of_shadow - 1)
        self.smoke_shell = max(0, self.smoke_shell - 1)
        self.epic_adventure = max(0, self.epic_adventure - 1)
        self.maple_world_goddess_blessing = max(
            0, self.maple_world_goddess_blessing - 1
        )
        self.spyder_in_mirror = max(0, self.spyder_in_mirror - 1)
        self.dark_flare = max(0, self.dark_flare - 1)


class ShadowerCooltime(BaseModel):
    ultimate_dark_sight: int
    ready_to_die: int
    soul_contract: int
    restraint_ring: int
    weaponpuff_ring: int
    vail_of_shadow: int
    smoke_shell: int
    epic_adventure: int
    maple_world_goddess_blessing: int
    spyder_in_mirror: int
    dark_flare: int
    sonic_blow: int
    slash_shadow_formation: int
    incision: int

    def to_list(self):
        return [
            self.ultimate_dark_sight,
            self.ready_to_die,
            self.soul_contract,
            self.restraint_ring,
            self.weaponpuff_ring,
            self.vail_of_shadow,
            self.smoke_shell,
            self.epic_adventure,
            self.maple_world_goddess_blessing,
            self.spyder_in_mirror,
            self.dark_flare,
            self.sonic_blow,
            self.slash_shadow_formation,
            self.incision,
        ]

    def step(self):
        self.ultimate_dark_sight = max(0, self.ultimate_dark_sight - 1)
        self.ready_to_die = max(0, self.ready_to_die - 1)
        self.soul_contract = max(0, self.soul_contract - 1)
        self.restraint_ring = max(0, self.restraint_ring - 1)
        self.weaponpuff_ring = max(0, self.weaponpuff_ring - 1)
        self.vail_of_shadow = max(0, self.vail_of_shadow - 1)
        self.smoke_shell = max(0, self.smoke_shell - 1)
        self.epic_adventure = max(0, self.epic_adventure - 1)
        self.maple_world_goddess_blessing = max(
            0, self.maple_world_goddess_blessing - 1
        )
        self.spyder_in_mirror = max(0, self.spyder_in_mirror - 1)
        self.dark_flare = max(0, self.dark_flare - 1)
        self.sonic_blow = max(0, self.sonic_blow - 1)
        self.slash_shadow_formation = max(0, self.slash_shadow_formation - 1)
        self.incision = max(0, self.incision - 1)
