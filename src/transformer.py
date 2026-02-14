import asyncio
import pandas as pd
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.models.tab_transformer.config import TabTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from omegaconf import DictConfig  
import torch.serialization  

async def fetch_data():
    transport = AIOHTTPTransport(url="https://gnomad.broadinstitute.org/api")
    client = Client(transport=transport, fetch_schema_from_transport=True)

    query = gql(
        """
        query VariantsInGene {
          gene(gene_symbol: "BRCA1", reference_genome: GRCh38) {
            variants(dataset: gnomad_r4) {
              variant_id
              pos
              exome {
                ac
                ac_hemi
                ac_hom
                an
                af
              }
            }
          }
        }
        """
    )

    result = await client.execute_async(query)
    return result["gene"]["variants"]

def preprocess_data(variants):
    rows = []
    for v in variants:
        exome = v.get("exome")
        if exome:
            rows.append({
                "pos": v["pos"],
                "ac": exome.get("ac"),
                "ac_hemi": exome.get("ac_hemi"),
                "ac_hom": exome.get("ac_hom"),
                "an": exome.get("an"),
                "af": exome.get("af"),
            })

    df = pd.DataFrame(rows)
    df.dropna(inplace=True)

    df["label"] = df["ac_hom"].apply(lambda x: 1 if x > 0 else 0)

    df.drop(columns=["ac_hom"], inplace=True)

    return df

def train_transformer(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    data_config = DataConfig(
        target=["label"],
        continuous_cols=[col for col in df.columns if col != "label"],
        categorical_cols=[], 
    )

    model_config = TabTransformerConfig(
        task="classification", 
        metrics=["accuracy"],    
    )

    trainer_config = TrainerConfig(
        max_epochs=8,
        batch_size=64,
    )

    optimizer_config = OptimizerConfig()

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
    )

    model.fit(train=train_df, validation=test_df)

    predictions = model.predict(test_df)
    accuracy = accuracy_score(test_df["label"], predictions["prediction"])

    print(f"\n✅ TabTransformer Accuracy: {accuracy * 100:.2f}%")
    print("\n📊 Example Prediction:")
    print("Input:", test_df.iloc[0][:-1].to_dict())
    print("Predicted:", model.predict(test_df.iloc[0:1])["prediction"].values[0])
    print("Actual:   ", test_df.iloc[0]["label"])

async def main():
    variants = await fetch_data()  
    df = preprocess_data(variants)  

    torch.serialization.add_safe_globals([DictConfig])

    train_transformer(df)  

asyncio.run(main())
