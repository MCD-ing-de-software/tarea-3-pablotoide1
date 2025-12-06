import pandas as pd
import pandas.testing as pdt
import unittest

import sys
sys.path.append(r"C:\Users\pmendez\OneDrive\Cursos\Magister\4 Ingeniería de software\Tarea 3\tarea-3-pablotoide1")
#import src.data_cleaner


from src.data_cleaner import DataCleaner



def make_sample_df() -> pd.DataFrame:
    """Create a small DataFrame for testing.

    The DataFrame intentionally contains missing values, extra whitespace
    in a text column, and an obvious numeric outlier.
    """
    return pd.DataFrame(
        {
            "name": [" Alice ", "Bob", None, " Carol  "],
            "age": [25, 1, 35, 120],  # 120 is a likely outlier
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        }
    )


class TestDataCleaner(unittest.TestCase):
    """Test suite for DataCleaner class."""

    def test_example_trim_strings_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar DataFrames completos.
        
        Este test demuestra cómo usar pandas.testing.assert_frame_equal() para comparar
        DataFrames completos, lo cual es útil porque maneja correctamente los índices,
        tipos de datos y valores NaN de Pandas.
        """
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "Carol"],
            "age": [25, 30, 35]
        })
        cleaner = DataCleaner()
        
        result = cleaner.trim_strings(df, ["name"])
        
        # DataFrame esperado después de trim
        expected = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "age": [25, 30, 35]
        })
        
        # Usar pandas.testing.assert_frame_equal() para comparar DataFrames completos
        # Esto maneja correctamente índices, tipos y estructura de Pandas
        pdt.assert_frame_equal(result, expected)

    def test_example_drop_invalid_rows_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar Series.
        
        Este test demuestra cómo usar pandas.testing.assert_series_equal() para comparar
        Series completas, útil cuando queremos verificar que una columna completa tiene
        los valores esperados manteniendo los índices correctos.
        """
        df = pd.DataFrame({
            "name": ["Alice", None, "Bob"],
            "age": [25, 30, None],
            "city": ["SCL", "LPZ", "SCL"]
        })
        cleaner = DataCleaner()
        
        result = cleaner.drop_invalid_rows(df, ["name"])
        
        # Verificar que la columna 'name' ya no tiene valores faltantes
        # Los índices después de drop_invalid_rows son [0, 2] (se eliminó la fila 1)
        expected_name_series = pd.Series(["Alice", "Bob"], index=[0, 2], name="name")
        
        # Usar pandas.testing.assert_series_equal() para comparar Series completas
        # Esto verifica valores, índices y tipos correctamente
        pdt.assert_series_equal(result["name"], expected_name_series, check_names=True)


################################## TEST 1: #################################

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        """Test que verifica que el método drop_invalid_rows elimina correctamente las filas
        que contienen valores faltantes (NaN o None) en las columnas especificadas.
        
        Escenario esperado:
        - Crear un DataFrame con valores faltantes usando make_sample_df()
        - Llamar a drop_invalid_rows con las columnas "name" y "age"
        - Verificar que el DataFrame resultante no tiene valores faltantes en esas columnas (usar self.assertEqual para comparar .isna().sum() con 0 - comparación simple de enteros, unittest es suficiente)
        - Verificar que el DataFrame resultante tiene menos filas que el original (usar self.assertLess con len() - comparación simple de enteros, unittest es suficiente)
        """

        
        df= make_sample_df()

        cleaner = DataCleaner()

        result = cleaner.drop_invalid_rows(df, ["name","age"])

        numero=result["name"].isna().sum()+result["age"].isna().sum()

        filas_original = len(df)
        filas_nuevo = len(result)


        self.assertEqual(numero, 0)
        self.assertLess(filas_nuevo,filas_original)


        
       
################################## TEST 2: #################################

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        """Test que verifica que el método drop_invalid_rows lanza un KeyError cuando
        se llama con una columna que no existe en el DataFrame.
        
        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a drop_invalid_rows con una columna que no existe (ej: "does_not_exist")
        - Verificar que se lanza un KeyError (usar self.assertRaises)
        """
        df= make_sample_df()
        cleaner = DataCleaner()

        
        with self.assertRaises(KeyError) as ma:
            cleaner.drop_invalid_rows(df, ["name","age","Columna xxx"])            
        exc = ma.exception


        a=str(exc)
        b="Columns not found in DataFrame"

        
        self.assertRegex(a, b)


################################## TEST 3: #################################

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        """Test que verifica que el método trim_strings elimina correctamente los espacios
        en blanco al inicio y final de los valores en las columnas especificadas, sin modificar
        el DataFrame original ni las columnas no especificadas.
        
        Escenario esperado:
        - Crear un DataFrame con espacios en blanco usando make_sample_df()
        - Llamar a trim_strings con la columna "name"
        - Verificar que el DataFrame original no fue modificado (mantiene los espacios) (usar self.assertEqual para comparar valores específicos como strings individuales - unittest es suficiente para strings)
        - Verificar que en el DataFrame resultante los valores de "name" no tienen espacios al inicio/final (usar self.assertEqual para comparar valores específicos como strings individuales - unittest es suficiente)
        - Verificar que las columnas no especificadas (ej: "city") permanecen sin cambios (si comparas Series completas, usar pandas.testing.assert_series_equal() ya que maneja mejor los índices y tipos de Pandas; si comparas valores individuales, self.assertEqual es suficiente)
        """
        df= make_sample_df()
        cleaner = DataCleaner()

## Antes de quitar espacios, hay que ver qué campos tienen string: Se debe filtrar los nulos

        df2= cleaner.drop_invalid_rows(df, ["name"])
        
        df3 = cleaner.trim_strings(df2, ["name"])
        
        df0=make_sample_df()

        pdt.assert_frame_equal(df,df0)

        lista_nombres=df3["name"].tolist()


        self.assertListEqual(lista_nombres, [s.strip() for s in lista_nombres])


        df_inicial_sin_name=df2.drop(["name"], axis=1)
        df_final_sin_name=df3.drop(["name"], axis=1)
                             

        pdt.assert_frame_equal(df_inicial_sin_name, df_final_sin_name)



################################## TEST 4: #################################

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        """Test que verifica que el método trim_strings lanza un TypeError cuando
        se llama con una columna que no es de tipo string.
        
        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a trim_strings con una columna numérica (ej: "age")
        - Verificar que se lanza un TypeError (usar self.assertRaises)
        """



        df= make_sample_df()
        cleaner = DataCleaner()

        
        with self.assertRaises(TypeError) as ma:
            cleaner.trim_strings(df, ["age"])            
        exc = ma.exception


        a=str(exc)
        b="Columns are not string dtype:"

        
        self.assertRegex(a, b)


################################## TEST 5: #################################


    def test_remove_outliers_iqr_removes_extreme_values(self):
        """Test que verifica que el método remove_outliers_iqr elimina correctamente los
        valores extremos (outliers) de una columna numérica usando el método del rango
        intercuartílico (IQR).
        
        Escenario esperado:
        - Crear un DataFrame con valores extremos usando make_sample_df() (contiene edad=120)
        - Llamar a remove_outliers_iqr con la columna "age" y factor=1.5
        - Verificar que el valor extremo (120) fue eliminado del resultado (usar self.assertNotIn para verificar que 120 no está en los valores de la columna)
        - Verificar que al menos uno de los valores no extremos (25 o 35) permanece en el resultado (usar self.assertIn para verificar que está presente)
        """

        df= make_sample_df()
        cleaner = DataCleaner()
        df1=cleaner.remove_outliers_iqr(df, "age", 1.5)   
      
        lista_edades=df1['age'].tolist()

        self.assertNotIn(120,df1['age'].tolist())

        self.assertTrue((25 in lista_edades) or (35 in lista_edades))

################################## TEST 6: #################################

    def test_remove_outliers_iqr_raises_keyerror_for_missing_column(self):
        """Test que verifica que el método remove_outliers_iqr lanza un KeyError cuando
        se llama con una columna que no existe en el DataFrame.
        
        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a remove_outliers_iqr con una columna que no existe (ej: "salary")
        - Verificar que se lanza un KeyError (usar self.assertRaises)
        """

        df= make_sample_df()
        cleaner = DataCleaner()
    
        with self.assertRaises(KeyError) as ma:
            cleaner.remove_outliers_iqr(df, "Altura", 1.5)       
        exc = ma.exception

        a=str(exc)
        b="not found in DataFrame"

        
        self.assertRegex(a, b)

################################## TEST 7: #################################

    def test_remove_outliers_iqr_raises_typeerror_for_non_numeric_column(self):
        """Test que verifica que el método remove_outliers_iqr lanza un TypeError cuando
        se llama con una columna que no es de tipo numérico.
        
        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a remove_outliers_iqr con una columna de texto (ej: "city")
        - Verificar que se lanza un TypeError (usar self.assertRaises)
        """

        df= make_sample_df()
        cleaner = DataCleaner()


        
        with self.assertRaises(TypeError) as ma:
            cleaner.remove_outliers_iqr(df, "city", 1.5)       
        exc = ma.exception


        a=str(exc)
        b="must be numeric to compute IQR"

        
        self.assertRegex(a, b)


if __name__ == "__main__":
    unittest.main()
