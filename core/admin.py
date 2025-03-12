from django.contrib import admin
from .models import Profile, LastFace
# Register your models here.
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('id', 'first_name', 'last_name', 'phone', 'email', 'status','present')
    list_filter = ('status',)
    search_fields = ('first_name', 'last_name', 'email')

admin.site.register(Profile, ProfileAdmin)
admin.site.register(LastFace)