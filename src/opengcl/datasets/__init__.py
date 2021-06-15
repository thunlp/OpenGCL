from .base_graph import Graph
from .planetoid_dataset import PubMed, Cora, CiteSeer
from .tu_dataset import Graphs, MUTAG, PTC_MR, IMDB_BINARY, \
    IMDB_MULTI, REDDIT_BINARY
from .pyg_dataset import PyG, DBLP, Coauthor_CS, \
    Coauthor_Phy, WikiCS, Amazon_Computers, Amazon_Photo

datasetlist = [PubMed, Cora, CiteSeer, MUTAG,
               PTC_MR, IMDB_BINARY, IMDB_MULTI, REDDIT_BINARY,
               DBLP, Coauthor_CS, Coauthor_Phy, WikiCS,
               Amazon_Computers, Amazon_Photo]

datasetdict = {Cls.__name__.lower(): Cls for Cls in datasetlist}

