import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, FORGE, \
    OBSERVER, ROBOTICSFACILITY, EFFECT_CHRONOBOOSTENERGYCOST
from sc2.player import Bot, Computer
from sc2.unit import Unit
import cv2
import numpy as np
import random
import time
from pathlib import Path
import keras

HEADLESS = False


class ProtossBot(sc2.BotAI):

    def __init__(self, use_model=False):
        self.MAX_WORKERS = 50
        self.do_something_after = 0
        self.train_data = []
        self.use_model = use_model
        if use_model:
            model_file = "BasicCNN-30-epochs-0.0001-LR-4.2"
            print(f"Using CNN model {model_file}")
            self.model = keras.models.load_model(f"model/{model_file}")

    async def on_step(self, iteration):
        time = (self.state.game_loop/22.4) / 60
        await self.scout()
        await self.distribute_workers()
        await self.train_probes()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand(time)
        await self.build_cybernetics_core()
        await self.build_robotics_facility()
        await self.build_gateways()
        await self.build_stargates()
        await self.train_force()
        game_map = await self.draw_map()
        await self.attack(iteration, game_map)

    async def scout(self):
        for scout in self.units(OBSERVER).idle:
            location = self.random_location_variance(self.enemy_start_locations[0])
            await self.do(scout.move(location))

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]
        x += ((random.randrange(-20, 20)) / 100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20)) / 100) * enemy_start_location[1]
        x = min(max(0, x), self.game_info.map_size[0])
        y = min(max(0, y), self.game_info.map_size[1])
        return position.Point2(position.Pointlike((x, y)))

    async def draw_map(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # unit: [size, (b, g, r)]
        structures = {
            NEXUS: [15, (0, 255, 0)],
            STARGATE: [5, (255, 0, 0)],
            ROBOTICSFACILITY: [5, (215, 155, 0)],
            PYLON: [3, (20, 235, 0)],
            GATEWAY: [3, (200, 100, 0)],
            CYBERNETICSCORE: [3, (150, 150, 0)],
            ASSIMILATOR: [2, (55, 200, 0)],
        }
        army = {
            VOIDRAY: [3, (255, 100, 0)],
            PROBE: [1, (55, 200, 0)],
        }

        for unit_type in structures:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), structures[unit_type][0], structures[unit_type][1],
                           -1)

        main_base_names = ["nexus", "commandcenter", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)
            else:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)

        for enemy_unit in self.known_enemy_units.not_structure:
            worker_names = ["probe",
                            "scv",
                            "drone"]
            # if that unit is a PROBE, SCV, or DRONE... it's a worker
            pos = enemy_unit.position
            if enemy_unit.name.lower() in worker_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
            else:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        for unit_type in army:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), army[unit_type][0], army[unit_type][1], -1)

        line_max = 50
        mineral_ratio = min(self.minerals / 1500, 1.0)
        vespene_ratio = min(self.vespene / 1500, 1.0)
        population_ratio = min(self.supply_left / self.supply_cap, 1.0)
        plausible_supply = self.supply_cap / 200.0
        military_weight = min(self.units(VOIDRAY).amount / (self.supply_cap - self.supply_left + 1), 1.0)

        cv2.line(game_data, (0, 19), (int(line_max * military_weight), 19), (250, 250, 200), 3)
        cv2.line(game_data, (0, 15), (int(line_max * plausible_supply), 15), (220, 200, 200), 3)
        cv2.line(game_data, (0, 11), (int(line_max * population_ratio), 11), (150, 150, 150), 3)
        cv2.line(game_data, (0, 7), (int(line_max * vespene_ratio), 7), (210, 200, 0), 3)
        cv2.line(game_data, (0, 3), (int(line_max * mineral_ratio), 3), (0, 255, 25), 3)

        flipped = cv2.flip(game_data, 0)

        if not HEADLESS:
            resized = cv2.resize(flipped, dsize=None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Map', resized)
            cv2.waitKey(1)

        return flipped

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

    async def expand(self, time):
        nexuses = self.units(NEXUS).amount
        if 0 < nexuses < min(5, max(3, time)):
            if self.can_afford(NEXUS):
                await self.expand_now(NEXUS)

    async def build_gateways(self):
        if self.units(PYLON).ready.exists \
                and self.units(GATEWAY).ready.amount < 1 \
                and not self.already_pending(GATEWAY) \
                and self.can_afford(GATEWAY):
            await self.build(GATEWAY, near=self.units(PYLON).ready.random)

    async def build_cybernetics_core(self):
        if self.units(GATEWAY).ready.exists \
                and not self.units(CYBERNETICSCORE).ready.exists \
                and not self.already_pending(CYBERNETICSCORE) \
                and self.can_afford(CYBERNETICSCORE):
            await self.build(CYBERNETICSCORE, near=self.units(PYLON).ready.random)

    async def build_robotics_facility(self):
        if self.units(CYBERNETICSCORE).ready.exists \
                and not self.units(ROBOTICSFACILITY).ready.exists \
                and not self.already_pending(ROBOTICSFACILITY) \
                and self.can_afford(ROBOTICSFACILITY):
            await self.build(ROBOTICSFACILITY, near=self.units(PYLON).ready.random)

    async def build_stargates(self):
        if self.units(CYBERNETICSCORE).ready.exists \
                and self.units(STARGATE).ready.amount < self.units(NEXUS).ready.amount \
                and not self.already_pending(STARGATE) \
                and self.can_afford(STARGATE):
            await self.build(STARGATE, near=self.units(PYLON).ready.random)

    async def train_force(self):
        if self.units(STARGATE).ready.exists:
            for stargate in self.units(STARGATE).ready.noqueue:
                if self.can_afford(VOIDRAY) and self.supply_left >= 4:
                    await self.do(stargate.train(VOIDRAY))
                    await self.chrono_boost(stargate)

        if not self.units(OBSERVER).exists:
            robotics = self.units(ROBOTICSFACILITY).ready.noqueue
            if robotics.exists and self.can_afford(OBSERVER) and self.supply_left >= 1:
                await self.do(robotics.first.train(OBSERVER))
                await self.chrono_boost(robotics.first)

    async def attack(self, time, game_map):
        if self.units(VOIDRAY).idle.exists:

            if self.use_model:
                prediction = self.model.predict([game_map.reshape([-1, 176, 200, 3])])
                choice = np.argmax(prediction[0])
            else:
                choice = random.randrange(0, 4)

            choice_dict = {0: "No Attack!",
                           1: "Attack close to our nexus!",
                           2: "Attack Enemy Structure!",
                           3: "Attack Eneemy Start!"}
            print(f"Choice #{choice}: {choice_dict[choice]}")

            target = False

            if time > self.do_something_after:
                if choice == 0:
                    # no attack
                    wait = random.randrange(7, 100) / 100
                    self.do_something_after = time + wait

                elif choice == 1:
                    # attack_unit_closest_nexus
                    if self.known_enemy_units.exists and self.units(NEXUS).exists:
                        target = self.known_enemy_units.closest_to(self.units(NEXUS).first)

                elif choice == 2:
                    # attack enemy structures
                    if self.known_enemy_structures.exists:
                        target = random.choice(self.known_enemy_structures)

                elif choice == 3:
                    # attack enemy start location
                    target = self.enemy_start_locations[0]

                if target:
                    for vr in self.units(VOIDRAY).idle:
                        await self.do(vr.attack(target))

                y = np.zeros(4)
                y[choice] = 1
                self.train_data.append([y, game_map])

    def find_target(self):
        if self.known_enemy_units.not_structure.exists:
            return self.known_enemy_units.not_structure.first
        elif self.known_enemy_structures.exists:
            return self.known_enemy_structures.first
        else:
            return self.enemy_start_locations[0]

    async def chrono_boost(self, target: Unit):
        for nexus in self.units(NEXUS).ready:
            if await self.can_cast(nexus, EFFECT_CHRONOBOOSTENERGYCOST, target):
                await self.do(nexus(EFFECT_CHRONOBOOSTENERGYCOST, target))
                break


bot = ProtossBot(use_model=True)
players = [
    Bot(Race.Protoss, bot),
    Computer(Race.Terran, Difficulty.Medium)
]
result = run_game(maps.get("AbyssalReefLE"), players, realtime=False)

if result == Result.Victory:
    data_folder = Path("train_data").absolute()
    data_folder.mkdir(parents=True, exist_ok=True)
    data_file = data_folder / "{}.npy".format(str(int(time.time())))
    print(f"Saving training data to {data_file}")
    np.save(data_file, np.array(bot.train_data))
