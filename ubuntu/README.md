# installing ubuntu

## on a Toshiba Portégé R930 1FE

you may have problems with the EFI bootloader

solution is here :

https://ubuntuforums.org/showthread.php?t=2247186

From live installer, as a sudoer, mount the efi partition on hard drive:

```
mount /dev/sda1 /mnt
cd /mnt/EFI
mkdir Microsoft
mkdir Microsoft/Boot
cp /mnt/EFI/ubuntu/grubx64.efi /mnt/EFI/Microsoft/Boot/bootmfgw.efi
```

if EFI/BOOT already exists, save it :
```
mv /mnt/EFI/BOOT /mnt/EFI/BOOT_OLD
```

feed the BOOT directory with file coming from the ubunutu folder
```
mkdir /mnt/EFI/BOOT
cp /mnt/EFI/ubuntu/* /mnt/EFI/BOOT
mv /mnt/EFI/BOOT/grubx64.efi /mnt/EFI/BOOT/bootx64.efi
```

## Grub

how to start from a grub command line

https://askubuntu.com/questions/616811/gnu-grub-terminal-instead-of-ubuntu-login-screen
```
grub> ls 
(hd0) (hd0,gpt1) (hd0,gpt2) (cd0)
```
assuming your linux partition is on /dev/sda2:
```
grub> linux  (hd0,gpt2)/vmlinuz root=/dev/sda2
grub> initrd  (hd0,gtp2)/initrd.img
grub> boot
```
