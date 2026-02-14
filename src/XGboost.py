import asyncio
import pandas as pd
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb


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
                "variant_id": v["variant_id"],
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

    df.drop(columns=["variant_id", "ac_hom"], inplace=True)

    return df

def train_model(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🎯 XGBoost Accuracy: {accuracy * 100:.2f}%")

    return model, X_test, y_test
async def main():
    variants = await fetch_data()
    df = preprocess_data(variants)
    model, X_test, y_test = train_model(df)

    print("\n📊 Example prediction for first test sample:")
    print("Input:", X_test.iloc[0].to_dict())
    print("Predicted:", model.predict([X_test.iloc[0]])[0])
    print("Actual:   ", y_test.iloc[0])

asyncio.run(main())