import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STALKER
import random


class ProtossBot(sc2.BotAI):

    async def on_step(self, iteration):
        await self.distribute_workers()
        await self.train_probes()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.build_gateway()
        await self.build_cybernetics_core()
        await self.train_force()
        await self.attack()

    async def train_probes(self):
        for nexus in self.units(NEXUS).ready.idle:
            if self.can_afford(PROBE) \
                    and self.units(PROBE).amount < self.units(NEXUS).amount * 16 + self.units(ASSIMILATOR).amount * 3:
                await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
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

    async def expand(self):
        nexuses = self.units(NEXUS).amount
        if 0 < nexuses < 4 and self.can_afford(NEXUS):
            await self.expand_now(NEXUS)

    async def build_gateway(self):
        if self.units(PYLON).ready.exists \
                and self.units(GATEWAY).ready.amount < self.units(NEXUS).ready.amount * 3 \
                and not self.already_pending(GATEWAY) \
                and self.can_afford(GATEWAY):
            await self.build(GATEWAY, near=self.units(PYLON).ready.random)

    async def build_cybernetics_core(self):
        if self.units(GATEWAY).ready.exists \
                and not self.units(CYBERNETICSCORE).ready.exists \
                and not self.already_pending(CYBERNETICSCORE) \
                and self.can_afford(CYBERNETICSCORE):
            await self.build(CYBERNETICSCORE, near=self.units(PYLON).ready.random)

    async def train_force(self):
        if self.units(CYBERNETICSCORE).ready.exists:
            for gateway in self.units(GATEWAY).ready.noqueue:
                if self.can_afford(STALKER) and self.supply_left >= 2:
                    await self.do(gateway.train(STALKER))

    async def attack(self):
        if self.units(STALKER).amount > 15:
            for stalker in self.units(STALKER).idle:
                await self.do(stalker.attack(self.find_target()))

    def find_target(self):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]


players = [
    Bot(Race.Protoss, ProtossBot()),
    Computer(Race.Terran, Difficulty.MediumHard)
]
run_game(maps.get("AbyssalReefLE"), players, realtime=False)
