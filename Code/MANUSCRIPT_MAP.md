# Manuscript scope

| Manuscript component | Code folders |
| --- | --- |
| Study area and settlement inventories | Data_Preparation, Spatial_Framework |
| Sea-ice preprocessing and POLARIS | Preprocessing, Parameters, Connectivity |
| Overall connection opportunities and seasonal changes | Connectivity, Figures |
| Regional and ship-class disparities | Regional_and_Ship_Class_Analysis, Figures |
| Port endpoint coverage and location set covering | Port_Coverage, Figures |

The code archive contains the workflows supporting the submitted manuscript. Climate-driver attribution, ERA5/ORAS5 experiments, sensitivity experiments outside the manuscript, automated monitoring, email utilities and Word manuscript editing are excluded.

Several supporting scripts are retained because the final workflow reads their outputs: the successive spatial-baseline scripts establish the final departure mappings, and the land-component assignments from the earlier 15 km workflow are reused in the final 100 km port analysis. The polar-basemap utilities are imported by the manuscript plotting scripts.
