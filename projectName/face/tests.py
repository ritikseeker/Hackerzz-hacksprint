from face.models import Person

# Delete a specific record
Person.objects.filter(name='Shubham').delete()
