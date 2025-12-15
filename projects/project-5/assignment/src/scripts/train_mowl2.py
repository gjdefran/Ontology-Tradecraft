import mowl
mowl.init_jvm("5g")
from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import ELEmModule
from mowl.datasets import PathDataset
from mowl.evaluation import Evaluator, RankingEvaluator
from mowl.projection import TaxonomyProjector, Edge
from mowl.evaluation import SubsumptionEvaluator
from tqdm import trange, tqdm
import torch as th
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class SubsumptionDataset(PathDataset):
    @property
    def evaluation_classes(self):
        return self.classes, self.classes

class ELEmbeddings(EmbeddingELModel):
    """
    Implementation based on [kulmanov2019]_.

    The idea of this paper is to embed EL by modeling ontology classes as :math:`n`-dimensional \
    balls (:math:`n`-balls) and ontology object properties as transformations of those \
    :math:`n`-balls. For each of the normal forms, there is a distance function defined that will \
    work as loss functions in the optimization framework.
    """

    
    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0,
                 reg_norm=1,
                 learning_rate=0.001,
                 epochs=1000,
                 batch_size=4096 * 8,
                 model_filepath=None,
                 device='cpu'
                 ):
        super().__init__(dataset, embed_dim, batch_size, extended=True, model_filepath=model_filepath)

        self.margin = margin
        self.reg_norm = reg_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self._loaded = False
        self.extended = False
        self.init_module()



    def init_module(self):
        self.module = ELEmModule(
            len(self.class_index_dict),  # number of ontology classes
            len(self.object_property_index_dict),  # number of ontology object properties
            len(self.individual_index_dict),  # number of individuals
            embed_dim=self.embed_dim,
            margin=self.margin
        ).to(self.device)






    def train(self, epochs=None, validate_every=1):
        logger.warning('You are using the default training method. If you want to use a cutomized training method (e.g., different negative sampling, etc.), please reimplement the train method in a subclass.')

        points_per_dataset = {k: len(v) for k, v in self.training_datasets.items()}
        string = "Training datasets: \n"
        for k, v in points_per_dataset.items():
            string += f"\t{k}: {v}\n"

        logger.info(string)
            
        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        best_loss = float('inf')

        all_classes_ids = list(self.class_index_dict.values())
        all_inds_ids = list(self.individual_index_dict.values())
        
        if epochs is None:
            epochs = self.epochs
        
        for epoch in trange(epochs):
            self.module.train()

            train_loss = 0
            loss = 0

            for gci_name, gci_dataset in self.training_datasets.items():
                if len(gci_dataset) == 0:
                    continue

                loss += th.mean(self.module(gci_dataset[:], gci_name))
                if gci_name == "gci2":
                    idxs_for_negs = np.random.choice(all_classes_ids, size=len(gci_dataset), replace=True)
                    rand_index = th.tensor(idxs_for_negs).to(self.device)
                    data = gci_dataset[:]
                    neg_data = th.cat([data[:, :2], rand_index.unsqueeze(1)], dim=1)
                    loss += th.mean(self.module(neg_data, gci_name, neg=True))

                if gci_name == "object_property_assertion":
                    idxs_for_negs = np.random.choice(all_inds_ids, size=len(gci_dataset), replace=True)
                    rand_index = th.tensor(idxs_for_negs).to(self.device)
                    data = gci_dataset[:]
                    neg_data = th.cat([data[:, :2], rand_index.unsqueeze(1)], dim=1)
                    loss += th.mean(self.module(neg_data, gci_name, neg=True))
                    
            loss += self.module.regularization_loss()
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            loss = 0

            if (epoch + 1) % validate_every == 0:
                if self.dataset.validation is not None:
                    with th.no_grad():
                        self.module.eval()
                        valid_loss = 0
                        gci2_data = self.validation_datasets["gci2"][:]
                        loss = th.mean(self.module(gci2_data, "gci2"))
                        valid_loss += loss.detach().item()


                        if valid_loss < best_loss:
                            best_loss = valid_loss
                            th.save(self.module.state_dict(), self.model_filepath)
                    print(f'Epoch {epoch+1}: Train loss: {train_loss} Valid loss: {valid_loss}')
                else:
                    print(f'Epoch {epoch+1}: Train loss: {train_loss}')


 



    def eval_method(self, data):
        return self.module.gci2_loss(data)






    def get_embeddings(self):
        self.init_module()

        print('Load the best model', self.model_filepath)
        self.load_best_model()
                
        ent_embeds = {
            k: v for k, v in zip(self.class_index_dict.keys(),
                                 self.module.class_embed.weight.cpu().detach().numpy())}
        rel_embeds = {
            k: v for k, v in zip(self.object_property_index_dict.keys(),
                                 self.module.rel_embed.weight.cpu().detach().numpy())}
        if self.module.ind_embed is not None:
            ind_embeds = {
                k: v for k, v in zip(self.individual_index_dict.keys(),
                                     self.module.ind_embed.weight.cpu().detach().numpy())}
        else:
            ind_embeds = None
        return ent_embeds, rel_embeds, ind_embeds





    def load_best_model(self):
        self.init_module()
        self.module.load_state_dict(th.load(self.model_filepath))
        self.module.eval()

def main():
    dataset = SubsumptionDataset(
    ontology_path = "../train.ttl",
    validation_path = "../valid.ttl",
    testing_path = "../valid.ttl"
)

    model = ELEmbeddings(
        dataset = dataset,
        embed_dim=30,
        margin=0.02,
        reg_norm=1,
        learning_rate=0.001,
        epochs=500,
        batch_size=64,
        model_filepath="reports/elembeddings_best.pt",
        device='cpu'

    )

    model.train(epochs=500)
    

    
    evaluator = SubsumptionEvaluator(
        dataset=dataset,
        batch_size=64,
        device ='cpu'
    )

    evaluation_model = model.module
    metrics = evaluator.evaluate(
        evaluation_model = evaluation_model,
        testing_ontology = dataset.validation,
        filter_ontologies = [dataset.ontology], 
        mode = "head_centric"
        )   

    print("Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")

    with open("reports/mowl_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to reports/mowl_metrics.json")

    emb, _, _ = model.get_embeddings()
    emb = {k: th.from_numpy(v) for k, v in emb.items()}
    cos = th.nn.functional.cosine_similarity

    # Deflecting Prism rdfs:subClassOf Prism
    Deflecting_Prism = "https://www.commoncoreontologies.org/ont00000001"
    Prism = "https://www.commoncoreontologies.org/ont00000804"

    # Waste Management Artifact Function rdfs:subClassOf Service Artifact Function
    Waste_Management_Artifact_Function = "https://www.commoncoreontologies.org/ont00000036"
    Service_Artifact_Function = "https://www.commoncoreontologies.org/ont00001301"

    # Portion of Solid Propellant rdfs:subClassOf Portion of Propellant
    Portion_of_Solid_Propellant = "https://www.commoncoreontologies.org/ont00000043"
    Portion_of_Propellant = "https://www.commoncoreontologies.org/ont00000593"

    # Stirling Engine rdfs:subClassOf External Combustion Engine
    Stirling_Engine = "https://www.commoncoreontologies.org/ont00000054"
    External_Combustion_Engine = "https://www.commoncoreontologies.org/ont00000431"

    # Reaction Engine rdfs:subClassOf Engine
    Reaction_Engine = "https://www.commoncoreontologies.org/ont00000056"
    Engine = "https://www.commoncoreontologies.org/ont00000210"

    # Portion of Coolant rdfs:subClassOf Portion of Material
    Portion_of_Coolant = "https://www.commoncoreontologies.org/ont00000091"
    Portion_of_Material = "https://www.commoncoreontologies.org/ont00000457"

    # Portion of Liquid Oxygen rdfs:subClassOf Portion of Cryogenic Material
    Portion_of_Liquid_Oxygen = "https://www.commoncoreontologies.org/ont00000101"
    Portion_of_Cryogenic_Material = "https://www.commoncoreontologies.org/ont00000255"

    # Air Inlet rdfs:subClassOf Fluid Control Artifact
    Air_Inlet = "https://www.commoncoreontologies.org/ont00000111"
    Fluid_Control_Artifact = "https://www.commoncoreontologies.org/ont00000256"

    # Power Transformer rdfs:subClassOf Material Artifact
    Power_Transformer = "https://www.commoncoreontologies.org/ont00000130"
    Material_Artifact = "https://www.commoncoreontologies.org/ont00000995"

    # Large-Scale Rocket Launcher rdfs:subClassOf Rocket Launcher
    Large_Scale_Rocket_Launcher = "https://www.commoncoreontologies.org/ont00000167"
    Rocket_Launcher = "https://www.commoncoreontologies.org/ont00001173"

    # Engine rdfs:subClassOf Material Artifact
    Engine = "https://www.commoncoreontologies.org/ont00000210"
    # Material_Artifact already defined above

    # Reflecting Optical Telescope rdfs:subClassOf Optical Telescope
    Reflecting_Optical_Telescope = "https://www.commoncoreontologies.org/ont00000219"
    Optical_Telescope = "https://www.commoncoreontologies.org/ont00001009"

    # Diffraction Grating rdfs:subClassOf Optical Instrument
    Diffraction_Grating = "https://www.commoncoreontologies.org/ont00000230"
    Optical_Instrument = "https://www.commoncoreontologies.org/ont00000136"

    # Heat Sink rdfs:subClassOf Material Artifact
    Heat_Sink = "https://www.commoncoreontologies.org/ont00000243"
    # Material_Artifact already defined above

    # Shaft rdfs:subClassOf Material Artifact
    Shaft = "https://www.commoncoreontologies.org/ont00000249"
    # Material_Artifact already defined above

    # Nozzle rdfs:subClassOf Fluid Control Artifact
    Nozzle = "https://www.commoncoreontologies.org/ont00000252"
    # Fluid_Control_Artifact already defined above

    # Information Bearing Entity rdfs:subClassOf BFO_0000030
    Information_Bearing_Entity = "https://www.commoncoreontologies.org/ont00000253"
    BFO_0000030 = "http://purl.obolibrary.org/obo/BFO_0000030"

    # Communication Interference Artifact Function rdfs:subClassOf Artifact Function
    Communication_Interference_Artifact_Function = "https://www.commoncoreontologies.org/ont00000273"
    Artifact_Function = "https://www.commoncoreontologies.org/ont00000323"

    # Radar Imaging Artifact Function rdfs:subClassOf Imaging Artifact Function
    Radar_Imaging_Artifact_Function = "https://www.commoncoreontologies.org/ont00000289"
    Imaging_Artifact_Function = "https://www.commoncoreontologies.org/ont00000601"

    # Vehicle Transmission rdfs:subClassOf Power Transmission Artifact
    Vehicle_Transmission = "https://www.commoncoreontologies.org/ont00000301"
    Power_Transmission_Artifact = "https://www.commoncoreontologies.org/ont00000663"

    # Healthcare Artifact Function rdfs:subClassOf Service Artifact Function
    Healthcare_Artifact_Function = "https://www.commoncoreontologies.org/ont00000309"
    # Service_Artifact_Function already defined above

    # Electronic Signal Processing Artifact Function rdfs:subClassOf Signal Processing Artifact Function
    Electronic_Signal_Processing_Artifact_Function = "https://www.commoncoreontologies.org/ont00000317"
    Signal_Processing_Artifact_Function = "https://www.commoncoreontologies.org/ont00001218"

    # Radio Transponder rdfs:subClassOf Radio Communication Instrument
    Radio_Transponder = "https://www.commoncoreontologies.org/ont00000325"
    Radio_Communication_Instrument = "https://www.commoncoreontologies.org/ont00000903"

    # Fan rdfs:subClassOf Fluid Control Artifact
    Fan = "https://www.commoncoreontologies.org/ont00000338"
    # Fluid_Control_Artifact already defined above

    # Mortar rdfs:subClassOf Cannon
    Mortar = "https://www.commoncoreontologies.org/ont00000340"
    Cannon = "https://www.commoncoreontologies.org/ont00000268"

    # Research Artifact Function rdfs:subClassOf Artifact Function
    Research_Artifact_Function = "https://www.commoncoreontologies.org/ont00000353"
    # Artifact_Function already defined above

    # Combustion Chamber rdfs:subClassOf Material Artifact
    Combustion_Chamber = "https://www.commoncoreontologies.org/ont00000397"
    # Material_Artifact already defined above

    # Communication Reception Artifact Function rdfs:subClassOf Artifact Function
    Communication_Reception_Artifact_Function = "https://www.commoncoreontologies.org/ont00000422"
    # Artifact_Function already defined above

    # Armored Fighting Vehicle rdfs:subClassOf Ground Motor Vehicle
    Armored_Fighting_Vehicle = "https://www.commoncoreontologies.org/ont00000427"
    Ground_Motor_Vehicle = "https://www.commoncoreontologies.org/ont00000053"

    # External Combustion Engine rdfs:subClassOf Combustion Engine
    # External_Combustion_Engine already defined above
    Combustion_Engine = "https://www.commoncoreontologies.org/ont00000746"

    # Collimation Artifact Function rdfs:subClassOf Artifact Function
    Collimation_Artifact_Function = "https://www.commoncoreontologies.org/ont00000451"
    # Artifact_Function already defined above

    # Railway rdfs:subClassOf Land Transportation Artifact
    Railway = "https://www.commoncoreontologies.org/ont00000456"
    Land_Transportation_Artifact = "https://www.commoncoreontologies.org/ont00001326"

    # Orientation Control Artifact Function rdfs:subClassOf Artifact Function
    Orientation_Control_Artifact_Function = "https://www.commoncoreontologies.org/ont00000466"
    # Artifact_Function already defined above

    # Material Copy of a Code List rdfs:subClassOf Material Copy of a List
    Material_Copy_of_a_Code_List = "https://www.commoncoreontologies.org/ont00000493"
    Material_Copy_of_a_List = "https://www.commoncoreontologies.org/ont00000799"

    # Pneumatic Power Source rdfs:subClassOf Power Source
    Pneumatic_Power_Source = "https://www.commoncoreontologies.org/ont00000516"
    Power_Source = "https://www.commoncoreontologies.org/ont00000288"

    # Financial Instrument rdfs:subClassOf Material Artifact
    Financial_Instrument = "https://www.commoncoreontologies.org/ont00000537"
    # Material_Artifact already defined above

    # Defoliant Artifact Function rdfs:subClassOf Herbicide Artifact Function
    Defoliant_Artifact_Function = "https://www.commoncoreontologies.org/ont00000538"
    Herbicide_Artifact_Function = "https://www.commoncoreontologies.org/ont00000831"

    # Telescope rdfs:subClassOf Imaging Instrument
    Telescope = "https://www.commoncoreontologies.org/ont00000547"
    Imaging_Instrument = "https://www.commoncoreontologies.org/ont00000771"

    # Sensor rdfs:subClassOf Transducer
    Sensor = "https://www.commoncoreontologies.org/ont00000569"
    Transducer = "https://www.commoncoreontologies.org/ont00000736"

    # Material Copy of a Instrument Display Panel rdfs:subClassOf Information Bearing Artifact
    Material_Copy_of_a_Instrument_Display_Panel = "https://www.commoncoreontologies.org/ont00000597"
    Information_Bearing_Artifact = "https://www.commoncoreontologies.org/ont00000798"

    # Refraction Artifact Function rdfs:subClassOf Artifact Function
    Refraction_Artifact_Function = "https://www.commoncoreontologies.org/ont00000635"
    # Artifact_Function already defined above

    # Highway rdfs:subClassOf Road
    Highway = "https://www.commoncoreontologies.org/ont00000646"
    Road = "https://www.commoncoreontologies.org/ont00000247"

    # Ground Moving Target Indication Artifact Function rdfs:subClassOf Moving Target Indication Artifact Function
    Ground_Moving_Target_Indication_Artifact_Function = "https://www.commoncoreontologies.org/ont00000650"
    Moving_Target_Indication_Artifact_Function = "https://www.commoncoreontologies.org/ont00000214"

    # Radiological Weapon rdfs:subClassOf Weapon
    Radiological_Weapon = "https://www.commoncoreontologies.org/ont00000656"
    Weapon = "https://www.commoncoreontologies.org/ont00000445"

    # Government Artifact Function rdfs:subClassOf Service Artifact Function
    Government_Artifact_Function = "https://www.commoncoreontologies.org/ont00000661"
    # Service_Artifact_Function already defined above

    # Mirror rdfs:subClassOf Optical Instrument
    Mirror = "https://www.commoncoreontologies.org/ont00000682"
    # Optical_Instrument already defined above

    # Nozzle Mouth rdfs:subClassOf Fluid Control Artifact
    Nozzle_Mouth = "https://www.commoncoreontologies.org/ont00000694"
    # Fluid_Control_Artifact already defined above

    # Terminal Board rdfs:subClassOf Material Artifact
    Terminal_Board = "https://www.commoncoreontologies.org/ont00000706"
    # Material_Artifact already defined above

    # Visual Prosthesis rdfs:subClassOf Artificial Eye
    Visual_Prosthesis = "https://www.commoncoreontologies.org/ont00000709"
    Artificial_Eye = "https://www.commoncoreontologies.org/ont00001248"

    # Horn Antenna rdfs:subClassOf Radio Antenna
    Horn_Antenna = "https://www.commoncoreontologies.org/ont00000714"
    Radio_Antenna = "https://www.commoncoreontologies.org/ont00001192"

    # Communication Artifact Function rdfs:subClassOf Artifact Function
    Communication_Artifact_Function = "https://www.commoncoreontologies.org/ont00000727"
    # Artifact_Function already defined above

    # Fungicide Artifact Function rdfs:subClassOf Anti-Microbial Artifact Function
    Fungicide_Artifact_Function = "https://www.commoncoreontologies.org/ont00000762"
    Anti_Microbial_Artifact_Function = "https://www.commoncoreontologies.org/ont00000415"

    # Optical Focusing Artifact Function rdfs:subClassOf Optical Processing Artifact Function
    Optical_Focusing_Artifact_Function = "https://www.commoncoreontologies.org/ont00000764"
    Optical_Processing_Artifact_Function = "https://www.commoncoreontologies.org/ont00000504"

    # Video Camera rdfs:subClassOf Camera
    Video_Camera = "https://www.commoncoreontologies.org/ont00000774"
    Camera = "https://www.commoncoreontologies.org/ont00000199"

    # Locomotive rdfs:subClassOf Rail Transport Vehicle
    Locomotive = "https://www.commoncoreontologies.org/ont00000775"
    Rail_Transport_Vehicle = "https://www.commoncoreontologies.org/ont00000796"

    # Signal Detection Artifact Function rdfs:subClassOf Artifact Function
    Signal_Detection_Artifact_Function = "https://www.commoncoreontologies.org/ont00000776"
    # Artifact_Function already defined above

    # Chemical Reaction Artifact Function rdfs:subClassOf Artifact Function
    Chemical_Reaction_Artifact_Function = "https://www.commoncoreontologies.org/ont00000791"
    # Artifact_Function already defined above

    # Catadioptric Optical Telescope rdfs:subClassOf Optical Telescope
    Catadioptric_Optical_Telescope = "https://www.commoncoreontologies.org/ont00000792"
    # Optical_Telescope already defined above

    # Fuel Ventilation System rdfs:subClassOf Fluid Control Artifact
    Fuel_Ventilation_System = "https://www.commoncoreontologies.org/ont00000793"
    # Fluid_Control_Artifact already defined above

    # Rail Transport Vehicle rdfs:subClassOf Ground Vehicle
    # Rail_Transport_Vehicle already defined above
    Ground_Vehicle = "https://www.commoncoreontologies.org/ont00000618"

    # Portion of Liquid Nitrogen rdfs:subClassOf Portion of Cryogenic Material
    Portion_of_Liquid_Nitrogen = "https://www.commoncoreontologies.org/ont00000816"
    # Portion_of_Cryogenic_Material already defined above

    # Optical Communication Artifact Function rdfs:subClassOf Electromagnetic Communication Artifact Function
    Optical_Communication_Artifact_Function = "https://www.commoncoreontologies.org/ont00000823"
    Electromagnetic_Communication_Artifact_Function = "https://www.commoncoreontologies.org/ont00001148"

    # Telecommunication Network Line rdfs:subClassOf Communication Instrument
    Telecommunication_Network_Line = "https://www.commoncoreontologies.org/ont00000825"
    Communication_Instrument = "https://www.commoncoreontologies.org/ont00000346"

    # Portion of Fuel rdfs:subClassOf Portion of Material
    Portion_of_Fuel = "https://www.commoncoreontologies.org/ont00000838"
    # Portion_of_Material already defined above

    # Light Machine Gun rdfs:subClassOf Long Gun
    Light_Machine_Gun = "https://www.commoncoreontologies.org/ont00000851"
    Long_Gun = "https://www.commoncoreontologies.org/ont00001104"

    # Public Safety Artifact Function rdfs:subClassOf Service Artifact Function
    Public_Safety_Artifact_Function = "https://www.commoncoreontologies.org/ont00000864"
    # Service_Artifact_Function already defined above

    # Optical Camera rdfs:subClassOf Camera
    Optical_Camera = "https://www.commoncoreontologies.org/ont00000869"
    # Camera already defined above

    # Material Copy of a Video rdfs:subClassOf Information Bearing Artifact
    Material_Copy_of_a_Video = "https://www.commoncoreontologies.org/ont00000874"
    # Information_Bearing_Artifact already defined above

    # Identification Friend or Foe Transponder rdfs:subClassOf Radio Transponder
    Identification_Friend_or_Foe_Transponder = "https://www.commoncoreontologies.org/ont00000878"
    # Radio_Transponder already defined above

    # Radio Communication Instrument rdfs:subClassOf Communication Instrument
    # Radio_Communication_Instrument already defined above
    # Communication_Instrument already defined above

    # Landline Telephone rdfs:subClassOf Telephone
    Landline_Telephone = "https://www.commoncoreontologies.org/ont00000933"
    Telephone = "https://www.commoncoreontologies.org/ont00000907"

    # Thermal Imaging Artifact Function rdfs:subClassOf Imaging Artifact Function
    Thermal_Imaging_Artifact_Function = "https://www.commoncoreontologies.org/ont00000944"
    # Imaging_Artifact_Function already defined above

    # Current Conversion Artifact Function rdfs:subClassOf Electrical Artifact Function
    Current_Conversion_Artifact_Function = "https://www.commoncoreontologies.org/ont00000961"
    Electrical_Artifact_Function = "https://www.commoncoreontologies.org/ont00000098"

    # Portion of Waste Material rdfs:subClassOf Portion of Processed Material
    Portion_of_Waste_Material = "https://www.commoncoreontologies.org/ont00000980"
    Portion_of_Processed_Material = "https://www.commoncoreontologies.org/ont00001084"

    # Reflective Prism rdfs:subClassOf Prism
    Reflective_Prism = "https://www.commoncoreontologies.org/ont00000988"
    # Prism already defined above

    # Power Rectifying Artifact Function rdfs:subClassOf Current Conversion Artifact Function
    Power_Rectifying_Artifact_Function = "https://www.commoncoreontologies.org/ont00000993"
    # Current_Conversion_Artifact_Function already defined above

    # Electrical Power Production Artifact Function rdfs:subClassOf Electrical Artifact Function
    Electrical_Power_Production_Artifact_Function = "https://www.commoncoreontologies.org/ont00000994"
    # Electrical_Artifact_Function already defined above

    # Material Artifact rdfs:subClassOf BFO_0000040
    # Material_Artifact already defined above
    BFO_0000040 = "http://purl.obolibrary.org/obo/BFO_0000040"

    # Controlled-Access Highway rdfs:subClassOf Highway
    Controlled_Access_Highway = "https://www.commoncoreontologies.org/ont00001015"
    # Highway already defined above

    # Equipment Cooling System rdfs:subClassOf Cooling System
    Equipment_Cooling_System = "https://www.commoncoreontologies.org/ont00001018"
    Cooling_System = "https://www.commoncoreontologies.org/ont00000350"

    # Train rdfs:subClassOf Rail Transport Vehicle
    Train = "https://www.commoncoreontologies.org/ont00001030"
    # Rail_Transport_Vehicle already defined above

    # Aircraft rdfs:subClassOf Vehicle
    Aircraft = "https://www.commoncoreontologies.org/ont00001043"
    Vehicle = "https://www.commoncoreontologies.org/ont00000713"

    # Artifact Model rdfs:subClassOf Artifact Design
    Artifact_Model = "https://www.commoncoreontologies.org/ont00001045"
    Artifact_Design = "https://www.commoncoreontologies.org/ont00000319"

    # Projectile Launcher rdfs:subClassOf Weapon
    Projectile_Launcher = "https://www.commoncoreontologies.org/ont00001051"
    # Weapon already defined above

    # Helical Antenna rdfs:subClassOf Wire Antenna
    Helical_Antenna = "https://www.commoncoreontologies.org/ont00001054"
    Wire_Antenna = "https://www.commoncoreontologies.org/ont00000499"

    # Air-Breathing Combustion Engine rdfs:subClassOf Reaction Engine
    Air_Breathing_Combustion_Engine = "https://www.commoncoreontologies.org/ont00001083"
    # Reaction_Engine already defined above

    # Orientation Observation Artifact Function rdfs:subClassOf Observation Artifact Function
    Orientation_Observation_Artifact_Function = "https://www.commoncoreontologies.org/ont00001087"
    Observation_Artifact_Function = "https://www.commoncoreontologies.org/ont00001306"

    # Electrical Power Storage Artifact Function rdfs:subClassOf Electrical Artifact Function
    Electrical_Power_Storage_Artifact_Function = "https://www.commoncoreontologies.org/ont00001092"
    # Electrical_Artifact_Function already defined above

    # Portion of Liquid Fuel rdfs:subClassOf Portion of Fuel
    Portion_of_Liquid_Fuel = "https://www.commoncoreontologies.org/ont00001135"
    # Portion_of_Fuel already defined above

    # Damaging Artifact Function rdfs:subClassOf Artifact Function
    Damaging_Artifact_Function = "https://www.commoncoreontologies.org/ont00001153"
    # Artifact_Function already defined above

    # Hydraulic Power Source rdfs:subClassOf Power Source
    Hydraulic_Power_Source = "https://www.commoncoreontologies.org/ont00001157"
    # Power_Source already defined above

    # Solvent Artifact Function rdfs:subClassOf Chemical Reaction Artifact Function
    Solvent_Artifact_Function = "https://www.commoncoreontologies.org/ont00001160"
    # Chemical_Reaction_Artifact_Function already defined above

    # Counterfeit Legal Instrument rdfs:subClassOf Counterfeit Instrument
    Counterfeit_Legal_Instrument = "https://www.commoncoreontologies.org/ont00001161"
    Counterfeit_Instrument = "https://www.commoncoreontologies.org/ont00000719"

    # Reaction Mass rdfs:subClassOf BFO_0000040
    Reaction_Mass = "https://www.commoncoreontologies.org/ont00001168"
    # BFO_0000040 already defined above

    # Rocket Launcher rdfs:subClassOf Projectile Launcher
    # Rocket_Launcher already defined above
    # Projectile_Launcher already defined above

    # Electromagnetic Shielding Artifact Function rdfs:subClassOf Artifact Function
    Electromagnetic_Shielding_Artifact_Function = "https://www.commoncoreontologies.org/ont00001174"
    # Artifact_Function already defined above

    # Retail Artifact Function rdfs:subClassOf Service Artifact Function
    Retail_Artifact_Function = "https://www.commoncoreontologies.org/ont00001210"
    # Service_Artifact_Function already defined above

    # Periscope rdfs:subClassOf Optical Instrument
    Periscope = "https://www.commoncoreontologies.org/ont00001216"
    # Optical_Instrument already defined above

    # Medium Machine Gun rdfs:subClassOf Mounted Gun
    Medium_Machine_Gun = "https://www.commoncoreontologies.org/ont00001217"
    Mounted_Gun = "https://www.commoncoreontologies.org/ont00000848"

    # Dish Receiver rdfs:subClassOf Radio Receiver
    Dish_Receiver = "https://www.commoncoreontologies.org/ont00001223"
    Radio_Receiver = "https://www.commoncoreontologies.org/ont00001145"

    # Alkaline Electric Battery rdfs:subClassOf Primary Cell Electric Battery
    Alkaline_Electric_Battery = "https://www.commoncoreontologies.org/ont00001229"
    Primary_Cell_Electric_Battery = "https://www.commoncoreontologies.org/ont00001050"

    # Sensor Artifact Function rdfs:subClassOf Artifact Function
    Sensor_Artifact_Function = "https://www.commoncoreontologies.org/ont00001241"
    # Artifact_Function already defined above

    # Cabin Pressurization Control System rdfs:subClassOf Environment Control System
    Cabin_Pressurization_Control_System = "https://www.commoncoreontologies.org/ont00001249"
    Environment_Control_System = "https://www.commoncoreontologies.org/ont00000453"

    # Material Copy of a Code 39 Barcode rdfs:subClassOf Material Copy of a One-Dimensional Barcode
    Material_Copy_of_a_Code_39_Barcode = "https://www.commoncoreontologies.org/ont00001260"
    Material_Copy_of_a_One_Dimensional_Barcode = "https://www.commoncoreontologies.org/ont00000258"

    # Material Copy of a Document rdfs:subClassOf Information Bearing Artifact
    Material_Copy_of_a_Document = "https://www.commoncoreontologies.org/ont00001298"
    # Information_Bearing_Artifact already defined above

    # Observation Artifact Function rdfs:subClassOf Artifact Function
    # Observation_Artifact_Function already defined above
    # Artifact_Function already defined above

    # Circuit Breaker rdfs:subClassOf Material Artifact
    Circuit_Breaker = "https://www.commoncoreontologies.org/ont00001343"
    # Material_Artifact already defined above

    # Legal Instrument rdfs:subClassOf Material Artifact
    Legal_Instrument = "https://www.commoncoreontologies.org/ont00001346"
    # Material_Artifact already defined above

    # Powering Artifact Function rdfs:subClassOf Artifact Function
    Powering_Artifact_Function = "https://www.commoncoreontologies.org/ont00001366"
    # Artifact_Function already defined above

    # Machine Bearing rdfs:subClassOf Material Artifact
    Machine_Bearing = "https://www.commoncoreontologies.org/ont00001370"
    # Material_Artifact already defined above

    # Bow rdfs:subClassOf Projectile Launcher
    Bow = "https://www.commoncoreontologies.org/ont00001373"
    # Projectile_Launcher already defined above

    # Electronic Cash rdfs:subClassOf Portion of Cash
    Electronic_Cash = "https://www.commoncoreontologies.org/ont00001382"
    Portion_of_Cash = "https://www.commoncoreontologies.org/ont00000475"

    print("sim(Deflecting Prism, Prism):", float(cos(emb[Deflecting_Prism], emb[Prism], dim=0)))
    print("sim(Waste Management Artifact Function, Service Artifact Function):", float(cos(emb[Waste_Management_Artifact_Function], emb[Service_Artifact_Function], dim=0)))
    print("sim(Portion of Solid Propellant, Portion of Propellant):", float(cos(emb[Portion_of_Solid_Propellant], emb[Portion_of_Propellant], dim=0)))
    print("sim(Stirling Engine, External Combustion Engine):", float(cos(emb[Stirling_Engine], emb[External_Combustion_Engine], dim=0)))
    print("sim(Reaction Engine, Engine):", float(cos(emb[Reaction_Engine], emb[Engine], dim=0)))
    print("sim(Portion of Coolant, Portion of Material):", float(cos(emb[Portion_of_Coolant], emb[Portion_of_Material], dim=0)))
    print("sim(Portion of Liquid Oxygen, Portion of Cryogenic Material):", float(cos(emb[Portion_of_Liquid_Oxygen], emb[Portion_of_Cryogenic_Material], dim=0)))
    print("sim(Air Inlet, Fluid Control Artifact):", float(cos(emb[Air_Inlet], emb[Fluid_Control_Artifact], dim=0)))
    print("sim(Power Transformer, Material Artifact):", float(cos(emb[Power_Transformer], emb[Material_Artifact], dim=0)))
    print("sim(Large-Scale Rocket Launcher, Rocket Launcher):", float(cos(emb[Large_Scale_Rocket_Launcher], emb[Rocket_Launcher], dim=0)))
    print("sim(Engine, Material Artifact):", float(cos(emb[Engine], emb[Material_Artifact], dim=0)))
    print("sim(Reflecting Optical Telescope, Optical Telescope):", float(cos(emb[Reflecting_Optical_Telescope], emb[Optical_Telescope], dim=0)))
    print("sim(Diffraction Grating, Optical Instrument):", float(cos(emb[Diffraction_Grating], emb[Optical_Instrument], dim=0)))
    print("sim(Heat Sink, Material Artifact):", float(cos(emb[Heat_Sink], emb[Material_Artifact], dim=0)))
    print("sim(Shaft, Material Artifact):", float(cos(emb[Shaft], emb[Material_Artifact], dim=0)))
    print("sim(Nozzle, Fluid Control Artifact):", float(cos(emb[Nozzle], emb[Fluid_Control_Artifact], dim=0)))
    print("sim(Information Bearing Entity, BFO_0000030):", float(cos(emb[Information_Bearing_Entity], emb[BFO_0000030], dim=0)))
    print("sim(Communication Interference Artifact Function, Artifact Function):", float(cos(emb[Communication_Interference_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Radar Imaging Artifact Function, Imaging Artifact Function):", float(cos(emb[Radar_Imaging_Artifact_Function], emb[Imaging_Artifact_Function], dim=0)))
    print("sim(Vehicle Transmission, Power Transmission Artifact):", float(cos(emb[Vehicle_Transmission], emb[Power_Transmission_Artifact], dim=0)))
    print("sim(Healthcare Artifact Function, Service Artifact Function):", float(cos(emb[Healthcare_Artifact_Function], emb[Service_Artifact_Function], dim=0)))
    print("sim(Electronic Signal Processing Artifact Function, Signal Processing Artifact Function):", float(cos(emb[Electronic_Signal_Processing_Artifact_Function], emb[Signal_Processing_Artifact_Function], dim=0)))
    print("sim(Radio Transponder, Radio Communication Instrument):", float(cos(emb[Radio_Transponder], emb[Radio_Communication_Instrument], dim=0)))
    print("sim(Fan, Fluid Control Artifact):", float(cos(emb[Fan], emb[Fluid_Control_Artifact], dim=0)))
    print("sim(Mortar, Cannon):", float(cos(emb[Mortar], emb[Cannon], dim=0)))
    print("sim(Research Artifact Function, Artifact Function):", float(cos(emb[Research_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Combustion Chamber, Material Artifact):", float(cos(emb[Combustion_Chamber], emb[Material_Artifact], dim=0)))
    print("sim(Communication Reception Artifact Function, Artifact Function):", float(cos(emb[Communication_Reception_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Armored Fighting Vehicle, Ground Motor Vehicle):", float(cos(emb[Armored_Fighting_Vehicle], emb[Ground_Motor_Vehicle], dim=0)))
    print("sim(External Combustion Engine, Combustion Engine):", float(cos(emb[External_Combustion_Engine], emb[Combustion_Engine], dim=0)))
    print("sim(Collimation Artifact Function, Artifact Function):", float(cos(emb[Collimation_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Railway, Land Transportation Artifact):", float(cos(emb[Railway], emb[Land_Transportation_Artifact], dim=0)))
    print("sim(Orientation Control Artifact Function, Artifact Function):", float(cos(emb[Orientation_Control_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Material Copy of a Code List, Material Copy of a List):", float(cos(emb[Material_Copy_of_a_Code_List], emb[Material_Copy_of_a_List], dim=0)))
    print("sim(Pneumatic Power Source, Power Source):", float(cos(emb[Pneumatic_Power_Source], emb[Power_Source], dim=0)))
    print("sim(Financial Instrument, Material Artifact):", float(cos(emb[Financial_Instrument], emb[Material_Artifact], dim=0)))
    print("sim(Defoliant Artifact Function, Herbicide Artifact Function):", float(cos(emb[Defoliant_Artifact_Function], emb[Herbicide_Artifact_Function], dim=0)))
    print("sim(Telescope, Imaging Instrument):", float(cos(emb[Telescope], emb[Imaging_Instrument], dim=0)))
    print("sim(Sensor, Transducer):", float(cos(emb[Sensor], emb[Transducer], dim=0)))
    print("sim(Material Copy of a Instrument Display Panel, Information Bearing Artifact):", float(cos(emb[Material_Copy_of_a_Instrument_Display_Panel], emb[Information_Bearing_Artifact], dim=0)))
    print("sim(Refraction Artifact Function, Artifact Function):", float(cos(emb[Refraction_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Highway, Road):", float(cos(emb[Highway], emb[Road], dim=0)))
    print("sim(Ground Moving Target Indication Artifact Function, Moving Target Indication Artifact Function):", float(cos(emb[Ground_Moving_Target_Indication_Artifact_Function], emb[Moving_Target_Indication_Artifact_Function], dim=0)))
    print("sim(Radiological Weapon, Weapon):", float(cos(emb[Radiological_Weapon], emb[Weapon], dim=0)))
    print("sim(Government Artifact Function, Service Artifact Function):", float(cos(emb[Government_Artifact_Function], emb[Service_Artifact_Function], dim=0)))
    print("sim(Mirror, Optical Instrument):", float(cos(emb[Mirror], emb[Optical_Instrument], dim=0)))
    print("sim(Nozzle Mouth, Fluid Control Artifact):", float(cos(emb[Nozzle_Mouth], emb[Fluid_Control_Artifact], dim=0)))
    print("sim(Terminal Board, Material Artifact):", float(cos(emb[Terminal_Board], emb[Material_Artifact], dim=0)))
    print("sim(Visual Prosthesis, Artificial Eye):", float(cos(emb[Visual_Prosthesis], emb[Artificial_Eye], dim=0)))
    print("sim(Horn Antenna, Radio Antenna):", float(cos(emb[Horn_Antenna], emb[Radio_Antenna], dim=0)))
    print("sim(Communication Artifact Function, Artifact Function):", float(cos(emb[Communication_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Fungicide Artifact Function, Anti-Microbial Artifact Function):", float(cos(emb[Fungicide_Artifact_Function], emb[Anti_Microbial_Artifact_Function], dim=0)))
    print("sim(Optical Focusing Artifact Function, Optical Processing Artifact Function):", float(cos(emb[Optical_Focusing_Artifact_Function], emb[Optical_Processing_Artifact_Function], dim=0)))
    print("sim(Video Camera, Camera):", float(cos(emb[Video_Camera], emb[Camera], dim=0)))
    print("sim(Locomotive, Rail Transport Vehicle):", float(cos(emb[Locomotive], emb[Rail_Transport_Vehicle], dim=0)))
    print("sim(Signal Detection Artifact Function, Artifact Function):", float(cos(emb[Signal_Detection_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Chemical Reaction Artifact Function, Artifact Function):", float(cos(emb[Chemical_Reaction_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Catadioptric Optical Telescope, Optical Telescope):", float(cos(emb[Catadioptric_Optical_Telescope], emb[Optical_Telescope], dim=0)))
    print("sim(Fuel Ventilation System, Fluid Control Artifact):", float(cos(emb[Fuel_Ventilation_System], emb[Fluid_Control_Artifact], dim=0)))
    print("sim(Rail Transport Vehicle, Ground Vehicle):", float(cos(emb[Rail_Transport_Vehicle], emb[Ground_Vehicle], dim=0)))
    print("sim(Portion of Liquid Nitrogen, Portion of Cryogenic Material):", float(cos(emb[Portion_of_Liquid_Nitrogen], emb[Portion_of_Cryogenic_Material], dim=0)))
    print("sim(Optical Communication Artifact Function, Electromagnetic Communication Artifact Function):", float(cos(emb[Optical_Communication_Artifact_Function], emb[Electromagnetic_Communication_Artifact_Function], dim=0)))
    print("sim(Telecommunication Network Line, Communication Instrument):", float(cos(emb[Telecommunication_Network_Line], emb[Communication_Instrument], dim=0)))
    print("sim(Portion of Fuel, Portion of Material):", float(cos(emb[Portion_of_Fuel], emb[Portion_of_Material], dim=0)))
    print("sim(Light Machine Gun, Long Gun):", float(cos(emb[Light_Machine_Gun], emb[Long_Gun], dim=0)))
    print("sim(Public Safety Artifact Function, Service Artifact Function):", float(cos(emb[Public_Safety_Artifact_Function], emb[Service_Artifact_Function], dim=0)))
    print("sim(Optical Camera, Camera):", float(cos(emb[Optical_Camera], emb[Camera], dim=0)))
    print("sim(Material Copy of a Video, Information Bearing Artifact):", float(cos(emb[Material_Copy_of_a_Video], emb[Information_Bearing_Artifact], dim=0)))
    print("sim(Identification Friend or Foe Transponder, Radio Transponder):", float(cos(emb[Identification_Friend_or_Foe_Transponder], emb[Radio_Transponder], dim=0)))
    print("sim(Radio Communication Instrument, Communication Instrument):", float(cos(emb[Radio_Communication_Instrument], emb[Communication_Instrument], dim=0)))
    print("sim(Landline Telephone, Telephone):", float(cos(emb[Landline_Telephone], emb[Telephone], dim=0)))
    print("sim(Thermal Imaging Artifact Function, Imaging Artifact Function):", float(cos(emb[Thermal_Imaging_Artifact_Function], emb[Imaging_Artifact_Function], dim=0)))
    print("sim(Current Conversion Artifact Function, Electrical Artifact Function):", float(cos(emb[Current_Conversion_Artifact_Function], emb[Electrical_Artifact_Function], dim=0)))
    print("sim(Portion of Waste Material, Portion of Processed Material):", float(cos(emb[Portion_of_Waste_Material], emb[Portion_of_Processed_Material], dim=0)))
    print("sim(Reflective Prism, Prism):", float(cos(emb[Reflective_Prism], emb[Prism], dim=0)))
    print("sim(Power Rectifying Artifact Function, Current Conversion Artifact Function):", float(cos(emb[Power_Rectifying_Artifact_Function], emb[Current_Conversion_Artifact_Function], dim=0)))
    print("sim(Electrical Power Production Artifact Function, Electrical Artifact Function):", float(cos(emb[Electrical_Power_Production_Artifact_Function], emb[Electrical_Artifact_Function], dim=0)))
    print("sim(Material Artifact, BFO_0000040):", float(cos(emb[Material_Artifact], emb[BFO_0000040], dim=0)))
    print("sim(Controlled-Access Highway, Highway):", float(cos(emb[Controlled_Access_Highway], emb[Highway], dim=0)))
    print("sim(Equipment Cooling System, Cooling System):", float(cos(emb[Equipment_Cooling_System], emb[Cooling_System], dim=0)))
    print("sim(Train, Rail Transport Vehicle):", float(cos(emb[Train], emb[Rail_Transport_Vehicle], dim=0)))
    print("sim(Aircraft, Vehicle):", float(cos(emb[Aircraft], emb[Vehicle], dim=0)))
    print("sim(Artifact Model, Artifact Design):", float(cos(emb[Artifact_Model], emb[Artifact_Design], dim=0)))
    print("sim(Projectile Launcher, Weapon):", float(cos(emb[Projectile_Launcher], emb[Weapon], dim=0)))
    print("sim(Helical Antenna, Wire Antenna):", float(cos(emb[Helical_Antenna], emb[Wire_Antenna], dim=0)))
    print("sim(Air-Breathing Combustion Engine, Reaction Engine):", float(cos(emb[Air_Breathing_Combustion_Engine], emb[Reaction_Engine], dim=0)))
    print("sim(Orientation Observation Artifact Function, Observation Artifact Function):", float(cos(emb[Orientation_Observation_Artifact_Function], emb[Observation_Artifact_Function], dim=0)))
    print("sim(Electrical Power Storage Artifact Function, Electrical Artifact Function):", float(cos(emb[Electrical_Power_Storage_Artifact_Function], emb[Electrical_Artifact_Function], dim=0)))
    print("sim(Portion of Liquid Fuel, Portion of Fuel):", float(cos(emb[Portion_of_Liquid_Fuel], emb[Portion_of_Fuel], dim=0)))
    print("sim(Damaging Artifact Function, Artifact Function):", float(cos(emb[Damaging_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Hydraulic Power Source, Power Source):", float(cos(emb[Hydraulic_Power_Source], emb[Power_Source], dim=0)))
    print("sim(Solvent Artifact Function, Chemical Reaction Artifact Function):", float(cos(emb[Solvent_Artifact_Function], emb[Chemical_Reaction_Artifact_Function], dim=0)))
    print("sim(Counterfeit Legal Instrument, Counterfeit Instrument):", float(cos(emb[Counterfeit_Legal_Instrument], emb[Counterfeit_Instrument], dim=0)))
    print("sim(Reaction Mass, BFO_0000040):", float(cos(emb[Reaction_Mass], emb[BFO_0000040], dim=0)))
    print("sim(Rocket Launcher, Projectile Launcher):", float(cos(emb[Rocket_Launcher], emb[Projectile_Launcher], dim=0)))
    print("sim(Electromagnetic Shielding Artifact Function, Artifact Function):", float(cos(emb[Electromagnetic_Shielding_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Retail Artifact Function, Service Artifact Function):", float(cos(emb[Retail_Artifact_Function], emb[Service_Artifact_Function], dim=0)))
    print("sim(Periscope, Optical Instrument):", float(cos(emb[Periscope], emb[Optical_Instrument], dim=0)))
    print("sim(Medium Machine Gun, Mounted Gun):", float(cos(emb[Medium_Machine_Gun], emb[Mounted_Gun], dim=0)))
    print("sim(Dish Receiver, Radio Receiver):", float(cos(emb[Dish_Receiver], emb[Radio_Receiver], dim=0)))
    print("sim(Alkaline Electric Battery, Primary Cell Electric Battery):", float(cos(emb[Alkaline_Electric_Battery], emb[Primary_Cell_Electric_Battery], dim=0)))
    print("sim(Sensor Artifact Function, Artifact Function):", float(cos(emb[Sensor_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Cabin Pressurization Control System, Environment Control System):", float(cos(emb[Cabin_Pressurization_Control_System], emb[Environment_Control_System], dim=0)))
    print("sim(Material Copy of a Code 39 Barcode, Material Copy of a One-Dimensional Barcode):", float(cos(emb[Material_Copy_of_a_Code_39_Barcode], emb[Material_Copy_of_a_One_Dimensional_Barcode], dim=0)))
    print("sim(Material Copy of a Document, Information Bearing Artifact):", float(cos(emb[Material_Copy_of_a_Document], emb[Information_Bearing_Artifact], dim=0)))
    print("sim(Observation Artifact Function, Artifact Function):", float(cos(emb[Observation_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Circuit Breaker, Material Artifact):", float(cos(emb[Circuit_Breaker], emb[Material_Artifact], dim=0)))
    print("sim(Legal Instrument, Material Artifact):", float(cos(emb[Legal_Instrument], emb[Material_Artifact], dim=0)))
    print("sim(Powering Artifact Function, Artifact Function):", float(cos(emb[Powering_Artifact_Function], emb[Artifact_Function], dim=0)))
    print("sim(Machine Bearing, Material Artifact):", float(cos(emb[Machine_Bearing], emb[Material_Artifact], dim=0)))
    print("sim(Bow, Projectile Launcher):", float(cos(emb[Bow], emb[Projectile_Launcher], dim=0)))
    print("sim(Electronic Cash, Portion of Cash):", float(cos(emb[Electronic_Cash], emb[Portion_of_Cash], dim=0)))

if __name__ == "__main__":
    main()
