use std::fs::File;

use polars::lazy::dsl::col;
use polars::prelude::*;
use polars_core::prelude::*;

use rustlearn::prelude::*;
use rustlearn::ensemble::random_forest::Hyperparameters;
use rustlearn::trees::decision_tree;

fn main() {
    let myschema = Schema::from_iter(
        vec![
            Field::new("Pclass", DataType::Float32),
            Field::new("Sex", DataType::String),
            Field::new("Sex_female", DataType::Float32),
            Field::new("Sex_male", DataType::Float32),
            Field::new("SibSp", DataType::Float32),
            Field::new("Parch", DataType::Float32),
        ]
    );

    let df_train = CsvReader::from_path("data/titanic/train.csv")
        .unwrap()
        .has_header(true)
        .with_dtypes(Some(Arc::new(myschema.clone())))
        .finish()
        .unwrap();

    let women = df_train
        .clone()
        .lazy()
        .filter(col("Sex").eq(lit("female")))
        .group_by([col("Survived")])
        .agg(
            vec![
                col("Survived").count().alias("count"),
            ]
        )
        .collect()
        .unwrap();

    let w = df_train
        .clone()
        .lazy()
        .filter(col("Sex").eq(lit("female")))
        .select([
            col("Survived").filter(col("Survived").eq(lit(1))).count().alias("Survived_count"),
            col("Survived").count().alias("Total_count"),
            (col("Survived").filter(col("Survived").eq(lit(1))).count().cast(DataType::Float64) / col("Survived").count().cast(DataType::Float64)).alias("survived_female_ratio"),
        ])
        .collect()
        .unwrap();
    let m = df_train
        .clone()
        .lazy()
        .filter(col("Sex").eq(lit("male")))
        .select([
            col("Survived").filter(col("Survived").eq(lit(1))).count().alias("Survived_count"),
            col("Survived").count().alias("Total_count"),
            (col("Survived").filter(col("Survived").eq(lit(1))).count().cast(DataType::Float64) / col("Survived").count().cast(DataType::Float64)).alias("survived_male_ratio"),
        ])
        .collect()
        .unwrap();

    let df_test= CsvReader::from_path("data/titanic/test.csv")
        .unwrap()
        .has_header(true)
        .with_dtypes(Some(Arc::new(myschema.clone())))
        .finish()
        .unwrap();

    let y = df_train["Survived"].cast(&DataType::Float32).unwrap();
    let features = vec!["Pclass", "Sex", "SibSp", "Parch"];
    let features_with_dummies = vec!["Pclass", "Sex_female", "Sex_male", "SibSp", "Parch"];
    
    let mut X: DataFrame = df_train.select(features.clone()).unwrap();
    X = X.columns_to_dummies(vec!["Sex"], None, false).unwrap();
    X = X
        .clone()
        .lazy()
        .with_column(
            col("Sex_female")
            .cast(DataType::Float32)
        )
        .with_column(
            col("Sex_male")
            .cast(DataType::Float32)
        )
        .collect()
        .unwrap();

    let mut X_test: DataFrame = df_test.select(features.clone()).unwrap();
    X_test = X_test.columns_to_dummies(vec!["Sex"], None, false).unwrap();
    X_test = X_test
        .clone()
        .lazy()
        .with_column(
            col("Sex_female")
            .cast(DataType::Float32)
        )
        .with_column(
            col("Sex_male")
            .cast(DataType::Float32)
        )
        .collect()
        .unwrap();

    let mut tree_params = decision_tree::Hyperparameters::new(X.shape().1 as usize);
    tree_params.min_samples_split(10).max_features(4);
    let mut model = Hyperparameters::new(tree_params, 10).one_vs_rest();

    let vec_X: Vec<Vec<f32>> = X
        .columns(features_with_dummies.clone())
        .unwrap()
        .iter()
        .map(|s|
            s
            .f32()
            .unwrap()
            .to_ndarray()
            .unwrap()
            .to_vec()
        ).collect();

    let vec_y = y.f32().unwrap().to_ndarray().unwrap().to_vec();

    let array_X = Array::from(&vec_X).T();
    let array_y = Array::from(vec_y);

    model.fit(&array_X, &array_y).unwrap();

    let vec_test_X: Vec<Vec<f32>> = X_test
        .columns(features_with_dummies)
        .unwrap()
        .iter()
        .map(|s|
            s
            .f32()
            .unwrap()
            .to_ndarray()
            .unwrap()
            .to_vec()
        ).collect();
    let array_test_X = Array::from(&vec_test_X).T();
    let predictions = model.predict(&array_test_X).unwrap();

    let mut df = DataFrame::new(vec![
        Series::new("PassengerId", (892..1310).collect::<Vec<_>>()),
        Series::new("Survived", predictions.data().iter().map(|x| x.round() as i32).collect::<Vec<i32>>()),
    ])
    .unwrap();

    let mut file = File::create("data/titanic/submission.csv").unwrap();
    CsvWriter::new(&mut file)
        .finish(&mut df)
        .unwrap();
        
}
