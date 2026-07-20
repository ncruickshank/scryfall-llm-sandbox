# packages

## docling
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    VlmPipelineOptions
)
from docling.datamodel.base_models import InputFormat
from docling.document_converter import ImageFormatOption

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    VlmPipelineOptions
)
from docling.datamodel.base_models import InputFormat
from docling.document_converter import ImageFormatOption

## other
from pathlib import Path
from tqdm import tqdm

# class
class DoclingCardTextDatasetGenerator():
    """
    Description
    ----------
    This class contains the necessary methods to convert a list of card image 
    paths into extracted plain text, and store the resulting file as a 
    new list of dicts (json-like object)

    NOTE: This class assumes we want to translate card images to text *one at a 
    time*. 

    Inputs
    ----------
    data = A list of dicts, where each dict is a collection of infomrmation about
        a card, critically including image_path
    use_granite = If true, we use the granite_docling version of the model instead
    """
    def __init__(
        self, 
        data:list[dict],
        use_granite:bool = False
    ):
        super().__init__()

        # store params as objects
        self.data = data 

        # instantiate objects for later use
        self.generated_text = [] # to be populated with dicts

        # instantiate the converter
        if use_granite:
            # configure VLM pipeline for granite
            pipeline_options = VlmPipelineOptions()
            pipeline_options.vlm_model = 'granite_docling'
            
            # set up the converter
            self.converter = DocumentConverter(
                format_options = {
                    InputFormat.IMAGE: ImageFormatOption(
                        pipeline_options = pipeline_options
                    )
                }
            )

            print('Using the "granite_docling" variant of docling.')
        else:
            # accept the default options
            self.converter = DocumentConverter()

    # === Main Methods ===

    def run(self):
        """
        Description
        ----------
        This public method iterates through each record in `self.data` to invoke
        the converter and generate the final output of text.

        Inputs
        ----------

        Returns
        ----------
        None, but self.generated_text will be populated in a similar structure to 
        self.data, except we have 'card_text_ocr' rather than Scryfall cleaned card text.
        """
        # define the progress bar
        for record in tqdm(self.data, desc = 'Reading card text from card images'):
            # invoke the converter
            gen_text = self._read_image(record = record)

            # define output
            out = {
                'id': record['id'],
                'oracle_id': record['oracle_id'],
                'card_name': record['card_name'],
                'card_text_ocr': gen_text,
                'tags': record['tags']
            }

            # store output
            self.generated_text.append(out)

    # === Internal Methods ===
    
    def _read_image(self, record:dict):
        """
        Description
        ----------
        This internal method is what gets iteratively called while looping through
        the `self.data` object to generate the requisite text.

        Inputs
        ----------
        record = The record from which we want to extract text

        Returns
        ----------

        """
        # define card path
        image_path = Path(record['image_path'])

        # run inference
        result = self.converter.convert(image_path)

        return result.document.export_to_text()