// main.js - JavaScript untuk aplikasi analisis curah hujan

document.addEventListener('DOMContentLoaded', function() {
    // Aktifkan semua tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Tambahkan kelas animate-fade-in ke semua card
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('animate-fade-in');
        }, index * 100);
    });
    
    // Fungsi untuk menampilkan loading indicator pada form submit
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitButton = this.querySelector('button[type="submit"]');
            if (submitButton) {
                const originalText = submitButton.innerHTML;
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="loading mr-2"></span> Memproses...';
                
                // Tampilkan kembali tombol submit setelah 30 detik jika halaman tidak teralihkan
                setTimeout(() => {
                    submitButton.disabled = false;
                    submitButton.innerHTML = originalText;
                }, 30000);
            }
        });
    });
    
    // Highlight active nav-link berdasarkan path URL saat ini
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath) {
            link.classList.add('active');
        }
    });
    
    // Fungsi untuk toggle collapsible elements
    const toggleButtons = document.querySelectorAll('[data-toggle="collapse"]');
    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                if (targetElement.classList.contains('show')) {
                    targetElement.classList.remove('show');
                } else {
                    targetElement.classList.add('show');
                }
            }
        });
    });
    
    // Fungsi untuk auto-hide alert messages setelah 5 detik
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        setTimeout(() => {
            const closeButton = alert.querySelector('.btn-close');
            if (closeButton) {
                closeButton.click();
            } else {
                alert.style.opacity = '0';
                setTimeout(() => {
                    alert.style.display = 'none';
                }, 500);
            }
        }, 5000);
    });
    
    // Fungsi untuk file upload preview
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const fileName = this.files[0]?.name;
            const fileLabel = this.nextElementSibling;
            if (fileLabel && fileName) {
                fileLabel.textContent = fileName;
            }
        });
    });
    
    // Fungsi untuk form validation
    const needsValidation = document.querySelectorAll('.needs-validation');
    Array.prototype.slice.call(needsValidation).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
    
    // Fungsi untuk filter table
    const tableFilter = document.getElementById('tableFilter');
    if (tableFilter) {
        tableFilter.addEventListener('keyup', function() {
            const filterValue = this.value.toLowerCase();
            const tableRows = document.querySelectorAll('.filterable-table tbody tr');
            
            tableRows.forEach(row => {
                const text = row.textContent.toLowerCase();
                if (text.indexOf(filterValue) > -1) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }
    
    // Fungsi untuk back-to-top button
    const backToTopBtn = document.getElementById('backToTopBtn');
    if (backToTopBtn) {
        window.onscroll = function() {
            if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) {
                backToTopBtn.style.display = 'block';
            } else {
                backToTopBtn.style.display = 'none';
            }
        };
        
        backToTopBtn.addEventListener('click', function() {
            document.body.scrollTop = 0; // For Safari
            document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
        });
    }
    
    // Fungsi untuk print halaman
    const printButtons = document.querySelectorAll('.btn-print');
    printButtons.forEach(button => {
        button.addEventListener('click', function() {
            window.print();
        });
    });
    
    // Animasi counter untuk angka statistik
    const counters = document.querySelectorAll('.counter');
    counters.forEach(counter => {
        const target = parseInt(counter.getAttribute('data-target'));
        const duration = 1000; // ms
        const step = target / (duration / 10);
        let current = 0;
        const counterInterval = setInterval(() => {
            current += step;
            counter.textContent = Math.round(current);
            if (current >= target) {
                counter.textContent = target;
                clearInterval(counterInterval);
            }
        }, 10);
    });
    
    // Fungsi untuk copy kode ke clipboard
    const copyButtons = document.querySelectorAll('.btn-copy-code');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const codeBlock = this.closest('.code-container').querySelector('code');
            const textArea = document.createElement('textarea');
            textArea.value = codeBlock.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            // Ubah teks tombol untuk konfirmasi
            const originalText = this.textContent;
            this.textContent = 'Disalin!';
            setTimeout(() => {
                this.textContent = originalText;
            }, 2000);
        });
    });
});

// Fungsi untuk memformat angka dengan pemisah ribuan
function formatNumber(number) {
    return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Fungsi untuk mengonversi tanggal ke format lokal Indonesia
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString('id-ID', options);
}

// Fungsi untuk dark mode toggle
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    const isDarkMode = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDarkMode ? 'enabled' : 'disabled');
}

// Periksa preferensi dark mode yang tersimpan
if (localStorage.getItem('darkMode') === 'enabled') {
    document.body.classList.add('dark-mode');
}