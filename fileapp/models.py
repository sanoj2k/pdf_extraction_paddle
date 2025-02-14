from django.db import models

class UploadedFile(models.Model):
    file_name = models.CharField(max_length=255, default="unknown.pdf")  # Store the file name
    uploaded_time = models.DateTimeField(auto_now_add=True)  # Store the upload time
    category = models.CharField(max_length=255, default="NA")  # Store the category

    def __str__(self):
        return f"{self.file_name} ({self.category})"
