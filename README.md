
---

## Graph Neural Network for Event Classification

In this study, we implement a Graph Neural Network (GNN) to classify signal and background events. The dataset consists of five event classes:

* **VBFHbb** (Signal)
* **QCD** (Background)
* **GGH** (Background)
* **ZJet** (Background)
* **TT** (Background)

---

## Graph Representation of Events

Each event is represented as a graph:

* The **8 leading jets (by pT)** in each event are selected.
* Each jet corresponds to a **node** in the graph.
* The graph therefore contains **8 nodes per event**.

To avoid **mass sculpting effects**, the Higgs candidate information will be excluded in one of the configurations.

---

## Node Features

Each node (jet) is characterized by the following features:

* Jet **kinematic four-vector**

  * Transverse momentum (pT)
  * Pseudorapidity (η)
  * Azimuthal angle (φ)
  * Energy (E)

* **b-tagging score**

* **Quark–Gluon Likelihood (QGL) score**

These features provide both kinematic and flavor-related information for classification.

---

## Graph Connectivity Strategies

We will investigate two different graph connectivity schemes:

### 1. Fully Connected Graph

* All nodes are connected to every other node.
* This allows the model to learn global correlations between all jets.

### 2. VBF-System Connectivity

* Only jets belonging to the **VBF system** are connected.
* This focuses on learning the specific topology characteristic of VBF events.

---

## Methodology

* A Graph Neural Network (GNN) will be trained to classify events into the five categories.
* The performance of different connectivity strategies will be compared.
* Special attention will be given to avoiding mass sculpting effects.

---

## Acknowledgement

We will begin implementation using a reference example from Kaggle and adapt it to our physics-specific event representation.


