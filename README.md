# Categorical-Feature-Representation-A-Survey-of-Approches-and-Applications
The repo is used to implement all mentioned representation approaches in the survey paper



```mermaid
flowchart TD
    Start([Start]) --> DataStructure{Is your data Tabular or Graph-based?}
    
    %% Tabular Data Path
    DataStructure -- Tabular --> CatVarType{Are your categorical variables Nominal or Ordinal?}
    CatVarType -- Nominal --> HighCardinality{Does your data have High Cardinality?}
    CatVarType -- Ordinal --> OrdinalMethods[Consider Ordinal Encoding or Target Encoding]
    OrdinalMethods --> HighCardinality
    
    HighCardinality -- Yes --> MemoryConstraint{Is Memory Efficiency Important?}
    HighCardinality -- No --> LowCardMethods[Consider One-Hot, Dummy, or Binary Encoding]
    
    MemoryConstraint -- Yes --> MemEffMethods[Consider Hash Encoding or CatBoost Encoding]
    MemoryConstraint -- No --> NonMemEffMethods[Consider Target Encoding or Leave-One-Out Encoding]
    
    %% Proceed to Supervision Decision
    MemEffMethods --> Supervised{Is your task Supervised or Unsupervised?}
    NonMemEffMethods --> Supervised
    LowCardMethods --> Supervised
    
    Supervised -- Supervised --> Dependency{Is there dependency with the Target Variable?}
    Supervised -- Unsupervised --> UnsupervisedMethods[Consider Similarity Encoding or Hash Encoding]
    
    Dependency -- Yes --> DependentMethods[Consider Target Encoding, CatBoost Encoding, WoE Encoding]
    Dependency -- No --> IndependentMethods[Consider Entity Embeddings or Word2Vec Embedding]
    
    %% Graph-Based Data Path
    DataStructure -- Graph-based --> Ontology{Does your graph incorporate an Ontology?}
    Ontology -- Yes --> OWL{Is the Ontology represented using OWL?}
    Ontology -- No --> NoOntologyMethods[Proceed to determine embedding focus]

    OWL -- Yes --> OWLMethods[Consider Owl2Vec*, opa2vec, Ontology2Vec]
    OWL -- No --> NonOWLMethods[Consider RDF2Vec or Graph2Vec]
    
    %% Determine Embedding Focus
    NoOntologyMethods & OWLMethods & NonOWLMethods --> EmbeddingFocus{What is the focus of your embedding?}
    EmbeddingFocus -- Node Embedding --> NodeMethods[Consider Node2Vec, catGCN, SemanticGraph2Vec]
    EmbeddingFocus -- Edge Embedding --> EdgeMethods[Consider Edge2Vec, Rank2Vec]
    EmbeddingFocus -- Subgraph Embedding --> SubgraphMethods[Consider Sub2Vec]
    EmbeddingFocus -- Graph-level Embedding --> GraphMethods[Consider Graph2Vec]
    
    %% Local vs Global Structure
    NodeMethods & EdgeMethods & SubgraphMethods & GraphMethods --> Structure{Are you interested in Local or Global Structure?}
    Structure -- Local Structure --> LocalMethods[Methods focusing on Local relationships]
    Structure -- Global Structure --> GlobalMethods[Methods capturing Global patterns]
    
    %% Automation and Scalability
    LocalMethods & GlobalMethods --> Automation{Do you require Automation and Scalability?}
    Automation -- Yes --> AutomatedMethods[For Tabular Data: Hash Encoding, CatBoost Encoding, Entity Embeddings;\nFor Graph Data: Node2Vec, Graph2Vec]
    Automation -- No --> ManualMethods[Methods requiring domain expertise:\nOntology2Vec, SCGE]
    
    %% Final Decision
    DependentMethods & IndependentMethods & UnsupervisedMethods & AutomatedMethods & ManualMethods --> SelectMethod[Select and Implement the Method]
    SelectMethod --> End([End])
