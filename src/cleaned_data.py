import asyncio
import pandas as pd
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport


async def fetch_gnomad_data(gene="BRCA1"):
    print(f"Fetching data for {gene}...")

    
    transport = AIOHTTPTransport(url="https://gnomad.broadinstitute.org/api")
    client = Client(transport=transport, fetch_schema_from_transport=True)

    
    query = gql(
        """
        query VariantsInGene {
          gene(gene_symbol: "%s", reference_genome: GRCh38) {
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
        """ % gene
    )

    
    result = await client.execute_async(query)

    
    variants = result["gene"]["variants"]
    df = pd.DataFrame(variants)

    
    df["ac"] = df["exome"].apply(lambda x: x["ac"] if x else None)
    df["ac_hemi"] = df["exome"].apply(lambda x: x["ac_hemi"] if x else None)
    df["ac_hom"] = df["exome"].apply(lambda x: x["ac_hom"] if x else None)
    df["an"] = df["exome"].apply(lambda x: x["an"] if x else None)
    df["af"] = df["exome"].apply(lambda x: x["af"] if x else None)

    
    df.drop(columns=["exome"], inplace=True)

    
    csv_filename = f"gnomad_{gene}_variants.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

    return df

async def main():
    gene = "BRCA1"  
    df = await fetch_gnomad_data(gene)
    
    if not df.empty:
        print(df.head(20))  

asyncio.run(main())