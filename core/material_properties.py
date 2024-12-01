class MaterialProperties:
    """
    Class to represent material properties.
    """

    def __init__(self, young_modulus: float, poisson_ratio: float):
        """
        Initialize the material properties.
        
        :param young_modulus: Young's modulus of the material (Pa).
        :param poisson_ratio: Poisson's ratio of the material (dimensionless).
        """
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio

    @property
    def shear_modulus(self) -> float:
        """
        Calculate the shear modulus of the material.
        
        :return: Shear modulus (Pa).
        """
        return self.young_modulus / (2 * (1 + self.poisson_ratio))

    @property
    def bulk_modulus(self) -> float:
        """
        Calculate the bulk modulus of the material.
        
        :return: Bulk modulus (Pa).
        """
        return self.young_modulus / (3 * (1 - 2 * self.poisson_ratio))

    @property
    def lambda_lame(self) -> float:
        """
        Calculate Lame's first parameter (lambda).
        
        :return: Lame's first parameter (Pa).
        """
        return (self.young_modulus * self.poisson_ratio) / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))


def calculate_material_properties(young_modulus: float, poisson_ratio: float) -> dict:
    """
    Function to calculate material properties as a dictionary.
    
    :param young_modulus: Young's modulus of the material (Pa).
    :param poisson_ratio: Poisson's ratio of the material (dimensionless).
    :return: A dictionary containing shear modulus, bulk modulus, and Lame's first parameter.
    """
    material = MaterialProperties(young_modulus, poisson_ratio)
    return {
        "shear_modulus": material.shear_modulus,
        "bulk_modulus": material.bulk_modulus,
        "lambda_lame": material.lambda_lame
    }
