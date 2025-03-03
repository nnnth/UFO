# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .coco import CocoDataset


# @DATASETS.register_module()
class LVISV05Dataset(CocoDataset):
    """LVIS v0.5 dataset for detection."""

    METAINFO = {
        'classes':
        ('acorn', 'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock',
         'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet',
         'antenna', 'apple', 'apple_juice', 'applesauce', 'apricot', 'apron',
         'aquarium', 'armband', 'armchair', 'armoire', 'armor', 'artichoke',
         'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award',
         'awning', 'ax', 'baby_buggy', 'basketball_backboard', 'backpack',
         'handbag', 'suitcase', 'bagel', 'bagpipe', 'baguet', 'bait', 'ball',
         'ballet_skirt', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage',
         'bandanna', 'banjo', 'banner', 'barbell', 'barge', 'barrel',
         'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat',
         'baseball_cap', 'baseball_glove', 'basket', 'basketball_hoop',
         'basketball', 'bass_horn', 'bat_(animal)', 'bath_mat', 'bath_towel',
         'bathrobe', 'bathtub', 'batter_(food)', 'battery', 'beachball',
         'bead', 'beaker', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed',
         'bedspread', 'cow', 'beef_(food)', 'beeper', 'beer_bottle',
         'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt', 'belt_buckle',
         'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor', 'binder',
         'binoculars', 'bird', 'birdfeeder', 'birdbath', 'birdcage',
         'birdhouse', 'birthday_cake', 'birthday_card', 'biscuit_(bread)',
         'pirate_flag', 'black_sheep', 'blackboard', 'blanket', 'blazer',
         'blender', 'blimp', 'blinker', 'blueberry', 'boar', 'gameboard',
         'boat', 'bobbin', 'bobby_pin', 'boiled_egg', 'bolo_tie', 'deadbolt',
         'bolt', 'bonnet', 'book', 'book_bag', 'bookcase', 'booklet',
         'bookmark', 'boom_microphone', 'boot', 'bottle', 'bottle_opener',
         'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)', 'bow-tie',
         'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'bowling_pin',
         'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere',
         'bread-bin', 'breechcloth', 'bridal_gown', 'briefcase',
         'bristle_brush', 'broccoli', 'broach', 'broom', 'brownie',
         'brussels_sprouts', 'bubble_gum', 'bucket', 'horse_buggy', 'bull',
         'bulldog', 'bulldozer', 'bullet_train', 'bulletin_board',
         'bulletproof_vest', 'bullhorn', 'corned_beef', 'bun', 'bunk_bed',
         'buoy', 'burrito', 'bus_(vehicle)', 'business_card', 'butcher_knife',
         'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car',
         'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf',
         'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)',
         'can', 'can_opener', 'candelabrum', 'candle', 'candle_holder',
         'candy_bar', 'candy_cane', 'walking_cane', 'canister', 'cannon',
         'canoe', 'cantaloup', 'canteen', 'cap_(headwear)', 'bottle_cap',
         'cape', 'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)',
         'elevator_car', 'car_battery', 'identity_card', 'card', 'cardigan',
         'cargo_ship', 'carnation', 'horse_carriage', 'carrot', 'tote_bag',
         'cart', 'carton', 'cash_register', 'casserole', 'cassette', 'cast',
         'cat', 'cauliflower', 'caviar', 'cayenne_(spice)', 'CD_player',
         'celery', 'cellular_telephone', 'chain_mail', 'chair',
         'chaise_longue', 'champagne', 'chandelier', 'chap', 'checkbook',
         'checkerboard', 'cherry', 'chessboard',
         'chest_of_drawers_(furniture)', 'chicken_(animal)', 'chicken_wire',
         'chickpea', 'Chihuahua', 'chili_(vegetable)', 'chime', 'chinaware',
         'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar',
         'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker',
         'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider',
         'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet',
         'clasp', 'cleansing_agent', 'clementine', 'clip', 'clipboard',
         'clock', 'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag',
         'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'coconut',
         'coffee_filter', 'coffee_maker', 'coffee_table', 'coffeepot', 'coil',
         'coin', 'colander', 'coleslaw', 'coloring_material',
         'combination_lock', 'pacifier', 'comic_book', 'computer_keyboard',
         'concrete_mixer', 'cone', 'control', 'convertible_(automobile)',
         'sofa_bed', 'cookie', 'cookie_jar', 'cooking_utensil',
         'cooler_(for_food)', 'cork_(bottle_plug)', 'corkboard', 'corkscrew',
         'edible_corn', 'cornbread', 'cornet', 'cornice', 'cornmeal', 'corset',
         'romaine_lettuce', 'costume', 'cougar', 'coverall', 'cowbell',
         'cowboy_hat', 'crab_(animal)', 'cracker', 'crape', 'crate', 'crayon',
         'cream_pitcher', 'credit_card', 'crescent_roll', 'crib', 'crock_pot',
         'crossbar', 'crouton', 'crow', 'crown', 'crucifix', 'cruise_ship',
         'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 'cube',
         'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupcake', 'hair_curler',
         'curling_iron', 'curtain', 'cushion', 'custard', 'cutting_tool',
         'cylinder', 'cymbal', 'dachshund', 'dagger', 'dartboard',
         'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk',
         'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table',
         'tux', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher',
         'dishwasher_detergent', 'diskette', 'dispenser', 'Dixie_cup', 'dog',
         'dog_collar', 'doll', 'dollar', 'dolphin', 'domestic_ass', 'eye_mask',
         'doorbell', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly',
         'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit',
         'dresser', 'drill', 'drinking_fountain', 'drone', 'dropper',
         'drum_(musical_instrument)', 'drumstick', 'duck', 'duckling',
         'duct_tape', 'duffel_bag', 'dumbbell', 'dumpster', 'dustpan',
         'Dutch_oven', 'eagle', 'earphone', 'earplug', 'earring', 'easel',
         'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater',
         'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk',
         'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan',
         'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)',
         'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)',
         'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose',
         'fireplace', 'fireplug', 'fish', 'fish_(food)', 'fishbowl',
         'fishing_boat', 'fishing_rod', 'flag', 'flagpole', 'flamingo',
         'flannel', 'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)',
         'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal',
         'folding_chair', 'food_processor', 'football_(American)',
         'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car',
         'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice',
         'fruit_salad', 'frying_pan', 'fudge', 'funnel', 'futon', 'gag',
         'garbage', 'garbage_truck', 'garden_hose', 'gargle', 'gargoyle',
         'garlic', 'gasmask', 'gazelle', 'gelatin', 'gemstone', 'giant_panda',
         'gift_wrap', 'ginger', 'giraffe', 'cincture',
         'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles',
         'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose',
         'gorilla', 'gourd', 'surgical_gown', 'grape', 'grasshopper', 'grater',
         'gravestone', 'gravy_boat', 'green_bean', 'green_onion', 'griddle',
         'grillroom', 'grinder_(tool)', 'grits', 'grizzly', 'grocery_bag',
         'guacamole', 'guitar', 'gull', 'gun', 'hair_spray', 'hairbrush',
         'hairnet', 'hairpin', 'ham', 'hamburger', 'hammer', 'hammock',
         'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel',
         'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw',
         'hardback_book', 'harmonium', 'hat', 'hatbox', 'hatch', 'veil',
         'headband', 'headboard', 'headlight', 'headscarf', 'headset',
         'headstall_(for_horses)', 'hearing_aid', 'heart', 'heater',
         'helicopter', 'helmet', 'heron', 'highchair', 'hinge', 'hippopotamus',
         'hockey_stick', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood',
         'hook', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce',
         'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear',
         'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
         'ice_tea', 'igniter', 'incense', 'inhaler', 'iPod',
         'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jean',
         'jeep', 'jelly_bean', 'jersey', 'jet_plane', 'jewelry', 'joystick',
         'jumpsuit', 'kayak', 'keg', 'kennel', 'kettle', 'key', 'keycard',
         'kilt', 'kimono', 'kitchen_sink', 'kitchen_table', 'kite', 'kitten',
         'kiwi_fruit', 'knee_pad', 'knife', 'knight_(chess_piece)',
         'knitting_needle', 'knob', 'knocker_(on_a_door)', 'koala', 'lab_coat',
         'ladder', 'ladle', 'ladybug', 'lamb_(animal)', 'lamb-chop', 'lamp',
         'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer',
         'lasagna', 'latch', 'lawn_mower', 'leather', 'legging_(clothing)',
         'Lego', 'lemon', 'lemonade', 'lettuce', 'license_plate', 'life_buoy',
         'life_jacket', 'lightbulb', 'lightning_rod', 'lime', 'limousine',
         'linen_paper', 'lion', 'lip_balm', 'lipstick', 'liquor', 'lizard',
         'Loafer_(type_of_shoe)', 'log', 'lollipop', 'lotion',
         'speaker_(stereo_equipment)', 'loveseat', 'machine_gun', 'magazine',
         'magnet', 'mail_slot', 'mailbox_(at_home)', 'mallet', 'mammoth',
         'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'martini',
         'mascot', 'mashed_potato', 'masher', 'mask', 'mast',
         'mat_(gym_equipment)', 'matchbox', 'mattress', 'measuring_cup',
         'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone',
         'microscope', 'microwave_oven', 'milestone', 'milk', 'minivan',
         'mint_candy', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money',
         'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor',
         'motor_scooter', 'motor_vehicle', 'motorboat', 'motorcycle',
         'mound_(baseball)', 'mouse_(animal_rodent)',
         'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom',
         'music_stool', 'musical_instrument', 'nailfile', 'nameplate',
         'napkin', 'neckerchief', 'necklace', 'necktie', 'needle', 'nest',
         'newsstand', 'nightshirt', 'nosebag_(for_animals)',
         'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker',
         'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil',
         'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'oregano',
         'ostrich', 'ottoman', 'overalls_(clothing)', 'owl', 'packet',
         'inkpad', 'pad', 'paddle', 'padlock', 'paintbox', 'paintbrush',
         'painting', 'pajamas', 'palette', 'pan_(for_cooking)',
         'pan_(metal_container)', 'pancake', 'pantyhose', 'papaya',
         'paperclip', 'paper_plate', 'paper_towel', 'paperback_book',
         'paperweight', 'parachute', 'parakeet', 'parasail_(sports)',
         'parchment', 'parka', 'parking_meter', 'parrot',
         'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport',
         'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter',
         'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'pegboard',
         'pelican', 'pen', 'pencil', 'pencil_box', 'pencil_sharpener',
         'pendulum', 'penguin', 'pennant', 'penny_(coin)', 'pepper',
         'pepper_mill', 'perfume', 'persimmon', 'baby', 'pet', 'petfood',
         'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano',
         'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow',
         'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball',
         'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)',
         'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat',
         'plate', 'platter', 'playing_card', 'playpen', 'pliers',
         'plow_(farm_equipment)', 'pocket_watch', 'pocketknife',
         'poker_(fire_stirring_tool)', 'pole', 'police_van', 'polo_shirt',
         'poncho', 'pony', 'pool_table', 'pop_(soda)', 'portrait',
         'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot',
         'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn',
         'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune',
         'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher',
         'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit',
         'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish',
         'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat',
         'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt',
         'recliner', 'record_player', 'red_cabbage', 'reflector',
         'remote_control', 'rhinoceros', 'rib_(food)', 'rifle', 'ring',
         'river_boat', 'road_map', 'robe', 'rocking_chair', 'roller_skate',
         'Rollerblade', 'rolling_pin', 'root_beer',
         'router_(computer_equipment)', 'rubber_band', 'runner_(carpet)',
         'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag',
         'safety_pin', 'sail', 'salad', 'salad_plate', 'salami',
         'salmon_(fish)', 'salmon_(food)', 'salsa', 'saltshaker',
         'sandal_(type_of_shoe)', 'sandwich', 'satchel', 'saucepan', 'saucer',
         'sausage', 'sawhorse', 'saxophone', 'scale_(measuring_instrument)',
         'scarecrow', 'scarf', 'school_bus', 'scissors', 'scoreboard',
         'scrambled_eggs', 'scraper', 'scratcher', 'screwdriver',
         'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane',
         'seashell', 'seedling', 'serving_dish', 'sewing_machine', 'shaker',
         'shampoo', 'shark', 'sharpener', 'Sharpie', 'shaver_(electric)',
         'shaving_cream', 'shawl', 'shears', 'sheep', 'shepherd_dog',
         'sherbert', 'shield', 'shirt', 'shoe', 'shopping_bag',
         'shopping_cart', 'short_pants', 'shot_glass', 'shoulder_bag',
         'shovel', 'shower_head', 'shower_curtain', 'shredder_(for_paper)',
         'sieve', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 'ski',
         'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'sled', 'sleeping_bag',
         'sling_(bandage)', 'slipper_(footwear)', 'smoothie', 'snake',
         'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock',
         'soda_fountain', 'carbonated_water', 'sofa', 'softball',
         'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon',
         'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
         'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'sponge',
         'spoon', 'sportswear', 'spotlight', 'squirrel',
         'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)',
         'steak_(food)', 'steak_knife', 'steamer_(kitchen_appliance)',
         'steering_wheel', 'stencil', 'stepladder', 'step_stool',
         'stereo_(sound_system)', 'stew', 'stirrer', 'stirrup',
         'stockings_(leg_wear)', 'stool', 'stop_sign', 'brake_light', 'stove',
         'strainer', 'strap', 'straw_(for_drinking)', 'strawberry',
         'street_sign', 'streetlight', 'string_cheese', 'stylus', 'subwoofer',
         'sugar_bowl', 'sugarcane_(plant)', 'suit_(clothing)', 'sunflower',
         'sunglasses', 'sunhat', 'sunscreen', 'surfboard', 'sushi', 'mop',
         'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato',
         'swimsuit', 'sword', 'syringe', 'Tabasco_sauce', 'table-tennis_table',
         'table', 'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag',
         'taillight', 'tambourine', 'army_tank', 'tank_(storage_vessel)',
         'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure',
         'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup',
         'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
         'telephone_pole', 'telephoto_lens', 'television_camera',
         'television_set', 'tennis_ball', 'tennis_racket', 'tequila',
         'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread',
         'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer',
         'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster',
         'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs',
         'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover',
         'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy',
         'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike',
         'trailer_truck', 'train_(railroad_vehicle)', 'trampoline', 'tray',
         'tree_house', 'trench_coat', 'triangle_(musical_instrument)',
         'tricycle', 'tripod', 'trousers', 'truck', 'truffle_(chocolate)',
         'trunk', 'vat', 'turban', 'turkey_(bird)', 'turkey_(food)', 'turnip',
         'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella',
         'underwear', 'unicycle', 'urinal', 'urn', 'vacuum_cleaner', 'valve',
         'vase', 'vending_machine', 'vent', 'videotape', 'vinegar', 'violin',
         'vodka', 'volleyball', 'vulture', 'waffle', 'waffle_iron', 'wagon',
         'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet',
         'walrus', 'wardrobe', 'wasabi', 'automatic_washer', 'watch',
         'water_bottle', 'water_cooler', 'water_faucet', 'water_filter',
         'water_heater', 'water_jug', 'water_gun', 'water_scooter',
         'water_ski', 'water_tower', 'watering_can', 'watermelon',
         'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit',
         'wheel', 'wheelchair', 'whipped_cream', 'whiskey', 'whistle', 'wick',
         'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
         'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket',
         'wineglass', 'wing_chair', 'blinder_(for_horses)', 'wok', 'wolf',
         'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht',
         'yak', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini'),
        'palette':
        None
    }

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        try:
            import lvis
            if getattr(lvis, '__version__', '0') >= '10.5.3':
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',  # noqa: E501
                    UserWarning)
            from lvis import LVIS
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'  # noqa: E501
            )
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.lvis = LVIS(local_path)
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.lvis.cat_img_map)

        img_ids = self.lvis.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.lvis.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            if raw_img_info['file_name'].startswith('COCO'):
                # Convert form the COCO 2014 file naming convention of
                # COCO_[train/val/test]2014_000000000000.jpg to the 2017
                # naming convention of 000000000000.jpg
                # (LVIS v1 will fix this naming issue)
                raw_img_info['file_name'] = raw_img_info['file_name'][-16:]
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.lvis.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.lvis

        return data_list


LVISDataset = LVISV05Dataset


@DATASETS.register_module()
class PACOLVISDataset(LVISDataset):
    """LVIS v1 dataset for detection."""

    METAINFO = {
        'classes':
        {23: 'trash_can', 35: 'handbag', 41: 'ball', 61: 'basket', 88: 'belt', 90: 'bench', 94: 'bicycle', 112: 'blender', 127: 'book', 133: 'bottle', 139: 'bowl', 
     143: 'box', 156: 'broom', 160: 'bucket', 184: 'calculator', 192: 'can', 207: 'car', 220: 'carton', 230: 'cellular_telephone', 232: 'chair', 271: 'clock', 324: 'crate', 
     344: 'cup', 378: 'dog', 396: 'drill', 399: 'drum', 409: 'earphone', 429: 'fan', 498: 'glass', 521: 'guitar', 530: 'hammer', 544: 'hat', 556: 'helmet', 591: 'jar', 
     604: 'kettle', 615: 'knife', 621: 'ladder', 626: 'lamp', 631: 'laptop_computer', 687: 'microwave_oven', 694: 'mirror', 705: 'mouse', 708: 'mug', 713: 'napkin', 
     719: 'newspaper', 751: 'pan', 781: 'pen', 782: 'pencil', 804: 'pillow', 811: 'pipe', 818: 'plate', 821: 'pliers', 881: 'remote_control', 898: 'plastic_bag', 
     921: 'scarf', 923: 'scissors', 926: 'screwdriver', 948: 'shoe', 973: 'slipper', 979: 'soap', 999: 'sponge', 1000: 'spoon', 1018: 'stool', 1042: 'sweater', 
     1050: 'table', 1061: 'tape', 1072: 'telephone', 1077: 'television_set', 1093: 'tissue_paper', 1108: 'towel', 1117: 'tray', 1139: 'vase', 1156: 'wallet', 1161: 'watch', 
     1196: 'wrench', 2000: ('car', 'antenna'), 2001: ('chair', 'apron'), 2002: ('table', 'apron'), 2003: ('chair', 'arm'), 2004: ('bench', 'arm'), 2005: ('chair', 'back'), 
     2006: ('guitar', 'back'), 2007: ('remote_control', 'back'), 2008: ('laptop_computer', 'back'), 2009: ('bench', 'back'), 2010: ('telephone', 'back_cover'), 
     2011: ('cellular_telephone', 'back_cover'), 2012: ('shoe', 'backstay'), 2013: ('belt', 'bar'), 2014: ('pen', 'barrel'), 2015: ('bottle', 'base'), 2016: ('bowl', 'base'), 
     2017: ('clock', 'base'), 2018: ('drum', 'base'), 2019: ('bucket', 'base'), 2020: ('handbag', 'base'), 2021: ('fan', 'base'), 2022: ('plate', 'base'), 2023: ('television_set', 'base'), 
     2024: ('basket', 'base'), 2025: ('can', 'base'), 2026: ('mug', 'base'), 2027: ('jar', 'base'), 2028: ('soap', 'base'), 2029: ('cup', 'base'), 2030: ('kettle', 'base'), 
     2031: ('tray', 'base'), 2032: ('chair', 'base'), 2033: ('pan', 'base'), 2034: ('blender', 'base'), 2035: ('lamp', 'base'), 2036: ('glass', 'base'), 2037: ('laptop_computer', 'base_panel'), 
     2038: ('bicycle', 'basket'), 2039: ('telephone', 'bezel'), 2040: ('cellular_telephone', 'bezel'), 2041: ('knife', 'blade'), 2042: ('fan', 'blade'), 2043: ('blender', 'blade'), 
     2044: ('pliers', 'blade'), 2045: ('scissors', 'blade'), 2046: ('plastic_bag', 'body'), 2047: ('bottle', 'body'), 2048: ('guitar', 'body'), 2049: ('bowl', 'body'), 2050: ('drill', 'body'), 
     2051: ('pencil', 'body'), 2052: ('drum', 'body'), 2053: ('sweater', 'body'), 2054: ('trash_can', 'body'), 2055: ('scarf', 'body'), 2056: ('bucket', 'body'), 2057: ('handbag', 'body'), 
     2058: ('plate', 'body'), 2059: ('calculator', 'body'), 2060: ('can', 'body'), 2061: ('mouse', 'body'), 2062: ('mug', 'body'), 2063: ('jar', 'body'), 2064: ('soap', 'body'), 
     2065: ('towel', 'body'), 2066: ('kettle', 'body'), 2067: ('glass', 'body'), 2068: ('vase', 'body'), 2069: ('dog', 'body'), 2070: ('towel', 'border'), 2071: ('can', 'bottom'), 
     2072: ('bucket', 'bottom'), 2073: ('handbag', 'bottom'), 2074: ('mug', 'bottom'), 2075: ('bottle', 'bottom'), 2076: ('crate', 'bottom'), 2077: ('bowl', 'bottom'), 2078: ('jar', 'bottom'), 
     2079: ('pan', 'bottom'), 2080: ('box', 'bottom'), 2081: ('plate', 'bottom'), 2082: ('soap', 'bottom'), 2083: ('tray', 'bottom'), 2084: ('glass', 'bottom'), 2085: ('television_set', 'bottom'), 
     2086: ('trash_can', 'bottom'), 2087: ('carton', 'bottom'), 2088: ('basket', 'bottom'), 2089: ('spoon', 'bowl'), 2090: ('fan', 'bracket'), 2091: ('guitar', 'bridge'), 2092: ('broom', 'brush'), 
     2093: ('broom', 'brush_cap'), 2094: ('belt', 'buckle'), 2095: ('watch', 'buckle'), 2096: ('lamp', 'bulb'), 2097: ('car', 'bumper'), 2098: ('telephone', 'button'), 2099: ('remote_control', 'button'), 
     2100: ('television_set', 'button'), 2101: ('cellular_telephone', 'button'), 2102: ('clock', 'cable'), 2103: ('blender', 'cable'), 2104: ('lamp', 'cable'), 2105: ('laptop_computer', 'cable'), 
     2106: ('earphone', 'cable'), 2107: ('kettle', 'cable'), 2108: ('laptop_computer', 'camera'), 2109: ('fan', 'canopy'), 2110: ('pen', 'cap'), 2111: ('soap', 'cap'), 2112: ('carton', 'cap'), 
     2113: ('bottle', 'cap'), 2114: ('soap', 'capsule'), 2115: ('bottle', 'capsule'), 2116: ('watch', 'case'), 2117: ('clock', 'case'), 2118: ('pen', 'clip'), 2119: ('soap', 'closure'), 
     2120: ('bottle', 'closure'), 2121: ('pipe', 'colied_tube'), 2122: ('microwave_oven', 'control_panel'), 2123: ('bucket', 'cover'), 2124: ('jar', 'cover'), 2125: ('blender', 'cover'), 
     2126: ('drum', 'cover'), 2127: ('basket', 'cover'), 2128: ('book', 'cover'), 2129: ('sweater', 'cuff'), 2130: ('blender', 'cup'), 2131: ('clock', 'decoration'), 2132: ('watch', 'dial'), 
     2133: ('microwave_oven', 'dial'), 2134: ('microwave_oven', 'door_handle'), 2135: ('bicycle', 'down_tube'), 2136: ('table', 'drawer'), 2137: ('mug', 'drawing'), 2138: ('dog', 'ear'), 
     2139: ('earphone', 'ear_pads'), 2140: ('pillow', 'embroidery'), 2141: ('belt', 'end_tip'), 2142: ('pencil', 'eraser'), 2143: ('dog', 'eye'), 2144: ('shoe', 'eyelet'), 2145: ('hammer', 'face'), 
     2146: ('helmet', 'face_shield'), 2147: ('fan', 'fan_box'), 2148: ('car', 'fender'), 2149: ('pencil', 'ferrule'), 2150: ('scissors', 'finger_hole'), 2151: ('guitar', 'fingerboard'), 
     2152: ('lamp', 'finial'), 2153: ('clock', 'finial'), 2154: ('wallet', 'flap'), 2155: ('blender', 'food_cup'), 2156: ('dog', 'foot'), 2157: ('vase', 'foot'), 2158: ('ladder', 'foot'), 
     2159: ('stool', 'footrest'), 2160: ('bicycle', 'fork'), 2161: ('belt', 'frame'), 2162: ('mirror', 'frame'), 2163: ('scarf', 'fringes'), 2164: ('bicycle', 'gear'), 2165: ('car', 'grille'), 
     2166: ('hammer', 'grip'), 2167: ('pen', 'grip'), 2168: ('watch', 'hand'), 2169: ('clock', 'hand'), 2170: ('plastic_bag', 'handle'), 2171: ('bottle', 'handle'), 2172: ('drill', 'handle'), 
     2173: ('hammer', 'handle'), 2174: ('scissors', 'handle'), 2175: ('broom', 'handle'), 2176: ('screwdriver', 'handle'), 2177: ('bucket', 'handle'), 2178: ('handbag', 'handle'), 
     2179: ('wrench', 'handle'), 2180: ('pliers', 'handle'), 2181: ('basket', 'handle'), 2182: ('mug', 'handle'), 2183: ('crate', 'handle'), 2184: ('jar', 'handle'), 2185: ('car', 'handle'), 
     2186: ('soap', 'handle'), 2187: ('cup', 'handle'), 2188: ('kettle', 'handle'), 2189: ('knife', 'handle'), 2190: ('spoon', 'handle'), 2191: ('pan', 'handle'), 2192: ('blender', 'handle'), 
     2193: ('vase', 'handle'), 2194: ('bicycle', 'handlebar'), 2195: ('dog', 'head'), 2196: ('hammer', 'head'), 2197: ('wrench', 'head'), 2198: ('drum', 'head'), 2199: ('bicycle', 'head_tube'), 
     2200: ('earphone', 'headband'), 2201: ('car', 'headlight'), 2202: ('guitar', 'headstock'), 2203: ('shoe', 'heel'), 2204: ('bottle', 'heel'), 2205: ('plastic_bag', 'hem'), 2206: ('sweater', 'hem'), 
     2207: ('towel', 'hem'), 2208: ('belt', 'hole'), 2209: ('guitar', 'hole'), 2210: ('trash_can', 'hole'), 2211: ('car', 'hood'), 2212: ('earphone', 'housing'), 2213: ('can', 'inner_body'), 
     2214: ('bucket', 'inner_body'), 2215: ('plastic_bag', 'inner_body'), 2216: ('handbag', 'inner_body'), 2217: ('mug', 'inner_body'), 2218: ('bottle', 'inner_body'), 2219: ('wallet', 'inner_body'), 
     2220: ('bowl', 'inner_body'), 2221: ('jar', 'inner_body'), 2222: ('blender', 'inner_body'), 2223: ('drum', 'inner_body'), 2224: ('glass', 'inner_body'), 2225: ('trash_can', 'inner_body'), 
     2226: ('cup', 'inner_body'), 2227: ('table', 'inner_body'), 2228: ('kettle', 'inner_body'), 2229: ('hat', 'inner_side'), 2230: ('crate', 'inner_side'), 2231: ('helmet', 'inner_side'), 
     2232: ('box', 'inner_side'), 2233: ('microwave_oven', 'inner_side'), 2234: ('pan', 'inner_side'), 2235: ('basket', 'inner_side'), 2236: ('carton', 'inner_side'), 2237: ('tray', 'inner_side'), 
     2238: ('plate', 'inner_wall'), 2239: ('tray', 'inner_wall'), 2240: ('shoe', 'insole'), 2241: ('slipper', 'insole'), 2242: ('pliers', 'jaw'), 2243: ('pliers', 'joint'), 2244: ('calculator', 'key'), 
     2245: ('guitar', 'key'), 2246: ('laptop_computer', 'keyboard'), 2247: ('soap', 'label'), 2248: ('bottle', 'label'), 2249: ('trash_can', 'label'), 2250: ('shoe', 'lace'), 2251: ('pencil', 'lead'), 
     2252: ('mouse', 'left_button'), 2253: ('table', 'leg'), 2254: ('bench', 'leg'), 2255: ('chair', 'leg'), 2256: ('dog', 'leg'), 2257: ('stool', 'leg'), 2258: ('can', 'lid'), 2259: ('crate', 'lid'), 
     2260: ('box', 'lid'), 2261: ('jar', 'lid'), 2262: ('pan', 'lid'), 2263: ('carton', 'lid'), 2264: ('trash_can', 'lid'), 2265: ('kettle', 'lid'), 2266: ('fan', 'light'), 2267: ('shoe', 'lining'), 
     2268: ('slipper', 'lining'), 2269: ('hat', 'logo'), 2270: ('mouse', 'logo'), 2271: ('helmet', 'logo'), 2272: ('fan', 'logo'), 2273: ('car', 'logo'), 2274: ('remote_control', 'logo'), 
     2275: ('laptop_computer', 'logo'), 2276: ('belt', 'loop'), 2277: ('bucket', 'loop'), 2278: ('drum', 'loop'), 2279: ('broom', 'lower_bristles'), 2280: ('watch', 'lug'), 2281: ('drum', 'lug'), 
     2282: ('car', 'mirror'), 2283: ('fan', 'motor'), 2284: ('vase', 'mouth'), 2285: ('bottle', 'neck'), 2286: ('spoon', 'neck'), 2287: ('vase', 'neck'), 2288: ('soap', 'neck'), 2289: ('dog', 'neck'), 
     2290: ('sweater', 'neckband'), 2291: ('dog', 'nose'), 2292: ('pipe', 'nozzle'), 2293: ('pipe', 'nozzle_stem'), 2294: ('tray', 'outer_side'), 2295: ('shoe', 'outsole'), 2296: ('slipper', 'outsole'), 
     2297: ('book', 'page'), 2298: ('bicycle', 'pedal'), 2299: ('trash_can', 'pedal'), 2300: ('fan', 'pedestal_column'), 2301: ('clock', 'pediment'), 2302: ('guitar', 'pickguard'), 2303: ('lamp', 'pipe'), 
     2304: ('hat', 'pom_pom'), 2305: ('belt', 'prong'), 2306: ('can', 'pull_tab'), 2307: ('soap', 'punt'), 2308: ('bottle', 'punt'), 2309: ('soap', 'push_pull_cap'), 2310: ('shoe', 'quarter'), 
     2311: ('chair', 'rail'), 2312: ('ladder', 'rail'), 2313: ('mouse', 'right_button'), 2314: ('can', 'rim'), 2315: ('bucket', 'rim'), 2316: ('hat', 'rim'), 2317: ('handbag', 'rim'), 2318: ('mug', 'rim'), 
     2319: ('bowl', 'rim'), 2320: ('helmet', 'rim'), 2321: ('jar', 'rim'), 2322: ('pan', 'rim'), 2323: ('car', 'rim'), 2324: ('plate', 'rim'), 2325: ('glass', 'rim'), 2326: ('tray', 'rim'), 
     2327: ('drum', 'rim'), 2328: ('trash_can', 'rim'), 2329: ('cup', 'rim'), 2330: ('table', 'rim'), 2331: ('basket', 'rim'), 2332: ('soap', 'ring'), 2333: ('broom', 'ring'), 2334: ('bottle', 'ring'), 
     2335: ('fan', 'rod'), 2336: ('tissue_paper', 'roll'), 2337: ('tape', 'roll'), 2338: ('car', 'roof'), 2339: ('sponge', 'rough_surface'), 2340: ('car', 'runningboard'), 2341: ('bicycle', 'saddle'), 
     2342: ('telephone', 'screen'), 2343: ('cellular_telephone', 'screen'), 2344: ('laptop_computer', 'screen'), 2345: ('scissors', 'screw'), 2346: ('mouse', 'scroll_wheel'), 2347: ('blender', 'seal_ring'), 
     2348: ('chair', 'seat'), 2349: ('bench', 'seat'), 2350: ('stool', 'seat'), 2351: ('car', 'seat'), 2352: ('bicycle', 'seat_stay'), 2353: ('bicycle', 'seat_tube'), 2354: ('lamp', 'shade'), 
     2355: ('lamp', 'shade_cap'), 2356: ('lamp', 'shade_inner_side'), 2357: ('broom', 'shaft'), 2358: ('screwdriver', 'shank'), 2359: ('table', 'shelf'), 2360: ('sweater', 'shoulder'), 
     2361: ('soap', 'shoulder'), 2362: ('bottle', 'shoulder'), 2363: ('crate', 'side'), 2364: ('guitar', 'side'), 2365: ('box', 'side'), 2366: ('microwave_oven', 'side'), 2367: ('pan', 'side'), 
     2368: ('television_set', 'side'), 2369: ('carton', 'side'), 2370: ('basket', 'side'), 2371: ('mouse', 'side_button'), 2372: ('car', 'sign'), 2373: ('soap', 'sipper'), 2374: ('bottle', 'sipper'), 
     2375: ('chair', 'skirt'), 2376: ('sweater', 'sleeve'), 2377: ('earphone', 'slider'), 2378: ('chair', 'spindle'), 2379: ('car', 'splashboard'), 2380: ('blender', 'spout'), 2381: ('soap', 'spout'), 
     2382: ('kettle', 'spout'), 2383: ('bottle', 'spout'), 2384: ('car', 'steeringwheel'), 2385: ('bicycle', 'stem'), 2386: ('stool', 'step'), 2387: ('ladder', 'step'), 2388: ('jar', 'sticker'), 
     2389: ('chair', 'stile'), 2390: ('hat', 'strap'), 2391: ('watch', 'strap'), 2392: ('helmet', 'strap'), 2393: ('slipper', 'strap'), 2394: ('belt', 'strap'), 2395: ('chair', 'stretcher'), 
     2396: ('table', 'stretcher'), 2397: ('bench', 'stretcher'), 2398: ('guitar', 'string'), 2399: ('fan', 'string'), 2400: ('blender', 'switch'), 2401: ('lamp', 'switch'), 2402: ('kettle', 'switch'), 
     2403: ('chair', 'swivel'), 2404: ('bench', 'table_top'), 2405: ('dog', 'tail'), 2406: ('car', 'taillight'), 2407: ('car', 'tank'), 2408: ('carton', 'tapering_top'), 2409: ('dog', 'teeth'), 
     2410: ('towel', 'terry_bar'), 2411: ('can', 'text'), 2412: ('plastic_bag', 'text'), 2413: ('mug', 'text'), 2414: ('newspaper', 'text'), 2415: ('jar', 'text'), 2416: ('carton', 'text'), 
     2417: ('shoe', 'throat'), 2418: ('microwave_oven', 'time_display'), 2419: ('spoon', 'tip'), 2420: ('pen', 'tip'), 2421: ('screwdriver', 'tip'), 2422: ('shoe', 'toe_box'), 2423: ('slipper', 'toe_box'), 
     2424: ('shoe', 'tongue'), 2425: ('bottle', 'top'), 2426: ('microwave_oven', 'top'), 2427: ('soap', 'top'), 2428: ('television_set', 'top'), 2429: ('table', 'top'), 2430: ('carton', 'top'), 
     2431: ('ladder', 'top_cap'), 2432: ('bicycle', 'top_tube'), 2433: ('laptop_computer', 'touchpad'), 2434: ('car', 'trunk'), 2435: ('car', 'turnsignal'), 2436: ('microwave_oven', 'turntable'), 
     2437: ('shoe', 'vamp'), 2438: ('slipper', 'vamp'), 2439: ('blender', 'vapour_cover'), 2440: ('hat', 'visor'), 2441: ('helmet', 'visor'), 2442: ('shoe', 'welt'), 2443: ('chair', 'wheel'), 
     2444: ('car', 'wheel'), 2445: ('trash_can', 'wheel'), 2446: ('table', 'wheel'), 2447: ('bicycle', 'wheel'), 2448: ('watch', 'window'), 2449: ('car', 'window'), 2450: ('car', 'windowpane'), 
     2451: ('car', 'windshield'), 2452: ('car', 'wiper'), 2453: ('mouse', 'wire'), 2454: ('sweater', 'yoke'), 2455: ('handbag', 'zip')},
        'palette':
        None
    }

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        try:
            import lvis
            if getattr(lvis, '__version__', '0') >= '10.5.3':
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',  # noqa: E501
                    UserWarning)
            from lvis import LVIS
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'  # noqa: E501
            )
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.lvis = LVIS(local_path)

        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.lvis.cat_img_map)

        img_ids = self.lvis.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.lvis.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            # coco_url is used in LVISv1 instead of file_name
            # e.g. http://images.cocodataset.org/train2017/000000391895.jpg
            # train/val split in specified in url
            raw_img_info['file_name'] = raw_img_info['coco_url'].replace(
                'http://images.cocodataset.org/', '')
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.lvis.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.lvis

        return data_list
