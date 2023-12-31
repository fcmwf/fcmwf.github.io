---
title: 一个等于号引发的惨案
tags: code
---

​

```c
uint64_t load_img(char *img_file){

// Lab2 TODO: load the 'img_file' to the start of pmem, and return its size
    FILE *file = fopen(img_file,"rb");
    if(file = NULL){
        printf("Failed to open file!\n");
        exit(1);
    }
    fseek(file,0,SEEK_END);
    long file_size = ftell(file);
    rewind(file);
    fread(pmem,1,file_size,file);
    fclose(file);
    return  file_size;
}
```

注没注意到`if(file = NULL)`
少了个等于号？没错，这个Bug我查了两个小时wwww
