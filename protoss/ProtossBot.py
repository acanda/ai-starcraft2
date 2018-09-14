import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, FORGE
from sc2.constants import UpgradeId, PROTOSSAIRWEAPONSLEVEL1, PROTOSSAIRARMORSLEVEL1, PROTOSSSHIELDSLEVEL1, \
    PROTOSSGROUNDWEAPONSLEVEL1, PROTOSSGROUNDARMORSLEVEL1
from sc2.constants import EFFECT_CHRONOBOOSTENERGYCOST
from sc2.player import Bot, Computer
from sc2.unit import Unit


class ProtossBot(sc2.BotAI):

    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 50

    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.train_probes()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand(iteration)
#        await self.build_forge()
        await self.build_cybernetics_core()
        await self.build_gateways(iteration)
        await self.build_stargates(iteration)
        await self.train_force()
        await self.attack()
        await self.research()

    async def train_probes(self):
        for nexus in self.units(NEXUS).ready.noqueue:
            if self.can_afford(PROBE) \
                    and self.units(PROBE).amount < self.MAX_WORKERS \
                    and self.units(PROBE).amount < self.units(NEXUS).amount * 16 + self.units(ASSIMILATOR).amount * 3:
                await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 10 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists and self.can_afford(PYLON):
                await self.build(PYLON, near=nexuses.first)

    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15, nexus)
            for vaspene in vaspenes:
                if self.can_afford(ASSIMILATOR) and not self.units(ASSIMILATOR).closer_than(1, vaspene).exists:
                    worker = self.select_build_worker(vaspene.position)
                    if worker is not None:
                        await self.do(worker.build(ASSIMILATOR, vaspene))

    async def expand(self, iteration):
        nexuses = self.units(NEXUS).amount
        if 0 < nexuses < min(5, max(3, iteration / self.ITERATIONS_PER_MINUTE)):
            if self.can_afford(NEXUS):
                await self.expand_now(NEXUS)

    async def build_forge(self):
        if self.units(PYLON).ready.exists \
                and not self.units(FORGE).ready.exists \
                and not self.already_pending(FORGE) \
                and self.can_afford(FORGE):
            await self.build(FORGE, near=self.units(PYLON).ready.random)

    async def build_gateways(self, iteration):
        if self.units(PYLON).ready.exists \
                and self.units(GATEWAY).ready.amount < self.units(NEXUS).ready.amount \
                and not self.already_pending(GATEWAY) \
                and self.can_afford(GATEWAY):
            await self.build(GATEWAY, near=self.units(PYLON).ready.random)

    async def build_cybernetics_core(self):
        if self.units(GATEWAY).ready.exists \
                and not self.units(CYBERNETICSCORE).ready.exists \
                and not self.already_pending(CYBERNETICSCORE) \
                and self.can_afford(CYBERNETICSCORE):
            await self.build(CYBERNETICSCORE, near=self.units(PYLON).ready.random)

    async def build_stargates(self, iteration):
        if self.units(CYBERNETICSCORE).ready.exists \
                and self.units(STARGATE).ready.amount < self.units(NEXUS).ready.amount \
                and not self.already_pending(STARGATE) \
                and self.can_afford(STARGATE):
            await self.build(STARGATE, near=self.units(PYLON).ready.random)

    async def train_force(self):
        if self.units(CYBERNETICSCORE).ready.exists:
            for gateway in self.units(GATEWAY).ready.noqueue:
                if self.can_afford(STALKER) and self.supply_left >= 2 \
                        and not self.units(STALKER).amount > self.units(VOIDRAY).amount:
                    await self.do(gateway.train(STALKER))
                    await self.chrono_boost(gateway)

        if self.units(STARGATE).ready.exists:
            for stargate in self.units(STARGATE).ready.noqueue:
                if self.can_afford(VOIDRAY) and self.supply_left >= 4:
                    await self.do(stargate.train(VOIDRAY))
                    await self.chrono_boost(stargate)

    async def attack(self):
        # { UNIT: [number to attack, number to defend] }
        attack_units = {
            STALKER: [15, 3],
            VOIDRAY: [8, 2]
        }

        for u in attack_units:
            idle_units = self.units(u).idle
            if idle_units.amount >= attack_units[u][0]:
                for attacker in idle_units:
                    await self.do(attacker.attack(self.find_target()))

            elif idle_units.amount >= attack_units[u][1] and self.known_enemy_units.amount > 0:
                close_enemies = self.known_enemy_units \
                    .filter(lambda enemy: self.units(NEXUS).ready.random.distance_to(enemy) < 25)
                if close_enemies.exists:
                    for attacker in idle_units:
                        await self.do(attacker.attack(close_enemies.random))

    def find_target(self):
        if len(self.known_enemy_units) > 0:
            return self.known_enemy_units.random
        elif len(self.known_enemy_structures) > 0:
            return self.known_enemy_structures.random
        else:
            return self.enemy_start_locations[0]

    async def research(self):
        for core in self.units(CYBERNETICSCORE).ready.noqueue:
            if self.can_research(PROTOSSAIRWEAPONSLEVEL1):
                await self.do(core.research(PROTOSSAIRWEAPONSLEVEL1))
                await self.chrono_boost(core)
            elif self.can_research(PROTOSSAIRARMORSLEVEL1):
                await self.do(core.research(PROTOSSAIRARMORSLEVEL1))
                await self.chrono_boost(core)

        for forge in self.units(FORGE).ready.noqueue:
            if self.can_research(PROTOSSSHIELDSLEVEL1):
                await self.do(forge.research(PROTOSSSHIELDSLEVEL1))
                await self.chrono_boost(forge)
            elif self.can_research(PROTOSSGROUNDWEAPONSLEVEL1):
                await self.do(forge.research(PROTOSSGROUNDWEAPONSLEVEL1))
                await self.chrono_boost(forge)
            elif self.can_research(PROTOSSGROUNDARMORSLEVEL1):
                await self.do(forge.research(PROTOSSGROUNDARMORSLEVEL1))
                await self.chrono_boost(forge)

    def can_research(self, upgrade_id: UpgradeId):
        return not self.already_pending_upgrade(upgrade_id) \
               and self.can_afford(upgrade_id)

    async def chrono_boost(self, target: Unit):
        for nexus in self.units(NEXUS).ready:
            if await self.can_cast(nexus, EFFECT_CHRONOBOOSTENERGYCOST, target):
                await self.do(nexus(EFFECT_CHRONOBOOSTENERGYCOST, target))
                break


players = [
    Bot(Race.Protoss, ProtossBot()),
    Computer(Race.Terran, Difficulty.Hard)
]
run_game(maps.get("AbyssalReefLE"), players, realtime=False)
