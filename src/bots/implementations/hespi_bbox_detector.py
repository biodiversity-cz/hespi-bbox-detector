from bots.base.abstract_url import AbstractUrlDatabot
from core.domain.DatabotRole import DatabotRole
from services.hespi_bbox_coco_service import HespiBboxCocoService


class HespiBboxDetectorDatabot(AbstractUrlDatabot):
    NAME = "hespi_bbox_detector"
    DESCRIPTION = (
        "Detect bounding boxes of herbarium sheet components "
        "(labels, stamps, handwriting, etc.) using a YOLO model "
        "derived from the hespi project. Output is in COCO JSON format."
    )
    VERSION = 1
    ROLE = DatabotRole.SCANNER

    def selectRecords(self) -> dict:
        return self.DATABASE.records_with_specimen(self.DB_ID, 50)

    def get_url(self, record: dict) -> str:
        return record["specimen_pid"]

    def fetch_data_from_url(self, url: str) -> dict:
        service = HespiBboxCocoService()
        return service.detect_from_url(url)
