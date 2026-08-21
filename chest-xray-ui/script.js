/**
 * ThoraxVision PACS-AI - Medical Radiology Workstation Controller
 * DenseNet121+GRU Multi-Label Diagnosis, Score-CAM XAI & Structured Expertise Sheet
 */

document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentImageBase64 = null;
    let currentPresetId = 'sample_xray';
    let currentFileName = 'sample_xray.jpg';
    let currentThreshold = 0.20;
    let currentPredictions = [];
    let currentXaiDisease = null;
    let isAnalyzing = false;
    let isRadiographInverted = false;
    let currentWlMode = 'default';
    let currentAlpha = 0.45;

    // Patient Metadata State (NIH ChestX-ray14 Dataset Standard)
    let currentPatient = {
        name: 'Subject #00013348',
        id: 'NIH-00013348_000',
        gender: '54Y / Male (PA View)'
    };

    // DOM Elements - Patient Bar
    const patientNameEl = document.getElementById('patient-name');
    const patientIdEl = document.getElementById('patient-id');
    const patientGenderEl = document.getElementById('patient-gender');
    const studyTimeEl = document.getElementById('study-time');

    // DOM Elements - Input & Controls
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdVal = document.getElementById('threshold-val');
    const dropzone = document.getElementById('dropzone');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const previewStage = document.getElementById('preview-stage');
    const previewImg = document.getElementById('preview-img');
    const fileNameLabel = document.getElementById('file-name-label');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const changeFileBtn = document.getElementById('change-file-btn');
    const resetUploadBtn = document.getElementById('reset-upload-btn');
    const invertBtn = document.getElementById('invert-btn');
    const wlButtons = document.querySelectorAll('.wl-btn');
    const presetCards = document.querySelectorAll('.preset-card');

    // Model Status Card Elements
    const statusIconBox = document.getElementById('status-icon-box');
    const statusMainIcon = document.getElementById('status-main-icon');
    const modelStatusTitle = document.getElementById('model-status-title');
    const modelStatusDesc = document.getElementById('model-status-desc');

    // Action CTA & Print
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyzeIcon = document.getElementById('analyze-icon');
    const analyzeText = document.getElementById('analyze-text');
    const printReportBtn = document.getElementById('print-report-btn');

    // Impression Banner
    const impressionBanner = document.getElementById('impression-banner');
    const impressionIcon = document.getElementById('impression-icon');
    const impressionStatusTitle = document.getElementById('impression-status-title');
    const triageTag = document.getElementById('triage-tag');
    const findingsChipsRow = document.getElementById('findings-chips-row');
    const clinicalRecNote = document.getElementById('clinical-recommendation-note');
    const recommendationTextContent = document.getElementById('recommendation-text-content');

    // Tabs
    const tabTriggers = document.querySelectorAll('.tab-trigger');
    const tabPanes = document.querySelectorAll('.tab-pane');

    // Metrics & List
    const metricDetected = document.getElementById('metric-detected');
    const metricTopProb = document.getElementById('metric-top-prob');
    const metricTopName = document.getElementById('metric-top-name');
    const diseaseList = document.getElementById('disease-list');

    // XAI Elements
    const xaiDiseaseSelect = document.getElementById('xai-disease-select');
    const xaiAlphaSlider = document.getElementById('xai-alpha-slider');
    const xaiAlphaLabel = document.getElementById('xai-alpha-label');
    const xaiTagProb = document.getElementById('xai-tag-prob');
    const xaiTagAuroc = document.getElementById('xai-tag-auroc');

    const showcaseImgOrig = document.getElementById('showcase-img-orig');
    const showcaseImgHeat = document.getElementById('showcase-img-heat');
    const overlayBaseImg = document.getElementById('overlay-base-img');
    const overlayHeatImg = document.getElementById('overlay-heat-img');
    const loadOrig = document.getElementById('load-orig');
    const loadHeat = document.getElementById('load-heat');
    const loadOver = document.getElementById('load-over');
    const interpretationDesc = document.getElementById('interpretation-desc');

    // Formal Expertise Sheet Elements (Tab 3)
    const expPName = document.getElementById('exp-p-name');
    const expPRm = document.getElementById('exp-p-rm');
    const expPGender = document.getElementById('exp-p-gender');
    const docRefNo = document.getElementById('doc-ref-no');

    const findingCor = document.getElementById('finding-cor');
    const findingPulmo = document.getElementById('finding-pulmo');
    const findingPleura = document.getElementById('finding-pleura');
    const findingBones = document.getElementById('finding-bones');

    const badgeCor = document.getElementById('badge-cor');
    const badgePulmo = document.getElementById('badge-pulmo');
    const badgePleura = document.getElementById('badge-pleura');
    const badgeBones = document.getElementById('badge-bones');
    const expertiseConclusion = document.getElementById('expertise-conclusion');

    // Toast
    const appToast = document.getElementById('app-toast');
    let toastTimer = null;

    // --- Theme Switcher Logic ---
    const themeMenuBtn = document.getElementById('theme-menu-btn');
    const themeDropdownMenu = document.getElementById('theme-dropdown-menu');
    const currentThemeLabel = document.getElementById('current-theme-label');
    const themeOptions = document.querySelectorAll('.theme-opt');

    const themeLabels = {
        'sage': 'Pastel Sage',
        'dark': 'Radiology Dark',
        'light': 'Clinical Light',
        'ocean': 'Ocean Teal'
    };

    function applyTheme(themeName) {
        document.body.setAttribute('data-theme', themeName);
        if (currentThemeLabel) {
            currentThemeLabel.textContent = themeLabels[themeName] || 'Pastel Sage';
        }
        themeOptions.forEach(opt => {
            if (opt.getAttribute('data-set-theme') === themeName) opt.classList.add('active');
            else opt.classList.remove('active');
        });
        localStorage.setItem('thorax_vision_theme', themeName);
    }

    if (themeMenuBtn && themeDropdownMenu) {
        themeMenuBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = themeDropdownMenu.style.display === 'flex';
            themeDropdownMenu.style.display = isOpen ? 'none' : 'flex';
        });

        document.addEventListener('click', (e) => {
            if (!themeDropdownMenu.contains(e.target) && e.target !== themeMenuBtn) {
                themeDropdownMenu.style.display = 'none';
            }
        });

        themeOptions.forEach(opt => {
            opt.addEventListener('click', (e) => {
                e.stopPropagation();
                const themeName = opt.getAttribute('data-set-theme');
                applyTheme(themeName);
                themeDropdownMenu.style.display = 'none';
                showToast(`Tema visual diubah: ${themeLabels[themeName]}`);
            });
        });

        const savedTheme = localStorage.getItem('thorax_vision_theme') || 'sage';
        applyTheme(savedTheme);
    }

    // --- Tab Switching ---
    tabTriggers.forEach(btn => {
        btn.addEventListener('click', () => {
            tabTriggers.forEach(t => t.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add('active');
        });
    });

    function navigateToTab(tabId) {
        tabTriggers.forEach(t => {
            if (t.getAttribute('data-tab') === tabId) t.classList.add('active');
            else t.classList.remove('active');
        });
        tabPanes.forEach(p => {
            if (p.id === tabId) p.classList.add('active');
            else p.classList.remove('active');
        });
    }

    // --- Threshold Slider (Real-time Re-ranking & Re-evaluation) ---
    if (thresholdSlider && thresholdVal) {
        thresholdSlider.addEventListener('input', (e) => {
            currentThreshold = parseFloat(e.target.value);
            thresholdVal.textContent = currentThreshold.toFixed(2);

            if (currentPredictions && currentPredictions.length > 0) {
                reEvaluateDiagnosis();
            }
        });
    }

    // --- Real-time Instant 0ms Overlay Blending Transparency Slider ---
    if (xaiAlphaSlider && xaiAlphaLabel) {
        xaiAlphaSlider.addEventListener('input', (e) => {
            currentAlpha = parseFloat(e.target.value);
            xaiAlphaLabel.textContent = `${Math.round(currentAlpha * 100)}%`;
            
            // Instant Hardware-Accelerated CSS Layered Blending (Zero Network Latency!)
            if (overlayHeatImg) {
                overlayHeatImg.style.opacity = currentAlpha;
            }
        });
    }

    // --- DICOM Viewport Window/Level (W/L) Presets ---
    wlButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            wlButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentWlMode = btn.getAttribute('data-wl');
            applyWlFilters();
        });
    });

    function applyWlFilters() {
        if (!previewImg) return;
        previewImg.classList.remove('wl-lung', 'wl-bone');
        if (currentWlMode === 'lung') previewImg.classList.add('wl-lung');
        if (currentWlMode === 'bone') previewImg.classList.add('wl-bone');
        showToast(`Windowing Mode: ${currentWlMode.toUpperCase()}`);
    }

    // --- Invert Colors & Radiograph Controls ---
    if (invertBtn && previewImg) {
        invertBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            isRadiographInverted = !isRadiographInverted;
            previewImg.classList.toggle('inverted', isRadiographInverted);
            showToast(isRadiographInverted ? 'LUT Inverted (Bones White)' : 'LUT Standar Aktif');
        });
    }

    if (browseBtn && fileInput) {
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (changeFileBtn && fileInput) {
        changeFileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (resetUploadBtn) {
        resetUploadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetUploadState();
        });
    }

    // --- Drag & Drop ---
    if (dropzone) {
        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('drag-active');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('drag-active');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt.files && dt.files.length > 0) {
                processUploadedFile(dt.files[0]);
            }
        });

        dropzone.addEventListener('click', () => {
            if (!currentImageBase64) {
                fileInput.click();
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                processUploadedFile(e.target.files[0]);
            }
        });
    }

    function processUploadedFile(file) {
        if (!file.type.match('image.*') && !file.name.endsWith('.dcm')) {
            showToast('Format berkas harus berupa citra (JPG, PNG, atau DICOM).');
            updateModelStatusCard('error', 'Gagal Membaca Citra', 'Format file tidak didukung. Harap unggah PNG, JPG, atau DICOM.');
            return;
        }

        currentFileName = file.name;
        currentPresetId = null;
        presetCards.forEach(c => c.classList.remove('active'));

        currentPatient = {
            name: `External Case (${file.name})`,
            id: `EXT-${Math.floor(100000 + Math.random() * 900000)}`,
            gender: 'Adult / Unspecified'
        };
        updatePatientDisplay();

        const reader = new FileReader();
        reader.onload = (e) => {
            currentImageBase64 = e.target.result;
            displayRadiographPreview(currentImageBase64, currentFileName);
            updateModelStatusCard('ready', 'Citra Siap Dianalisis', `Berkas '${currentFileName}' berhasil dimuat.`);
            showToast(`Citra radiografi dimuat: ${currentFileName}`);
        };
        reader.onerror = () => {
            updateModelStatusCard('error', 'Gagal Membaca File', 'Berkas citra rusak atau tidak dapat diuraikan.');
        };
        reader.readAsDataURL(file);
    }

    function displayRadiographPreview(src, name) {
        previewImg.src = src;
        fileNameLabel.textContent = name;
        dropzonePrompt.style.display = 'none';
        previewStage.style.display = 'flex';
        resetUploadBtn.style.display = 'inline-flex';
    }

    function resetUploadState() {
        currentImageBase64 = null;
        currentPresetId = null;
        currentFileName = '';
        fileInput.value = '';
        previewImg.src = '';
        dropzonePrompt.style.display = 'flex';
        previewStage.style.display = 'none';
        resetUploadBtn.style.display = 'none';
        presetCards.forEach(c => c.classList.remove('active'));
        updateModelStatusCard('ready', 'Status Model: Siap Inferensi', 'Menunggu instruksi analisis citra rontgen.');
    }

    // --- Preset Selection ---
    presetCards.forEach(card => {
        card.addEventListener('click', () => {
            const presetId = card.getAttribute('data-preset');
            currentPatient = {
                name: card.getAttribute('data-patient') || 'Pasien Anonim',
                id: card.getAttribute('data-id') || 'RM-2026-00000',
                gender: card.getAttribute('data-gender') || 'Laki-laki'
            };
            loadPresetCase(presetId);
        });
    });

    async function loadPresetCase(presetId) {
        presetCards.forEach(c => c.classList.remove('active'));
        const activeCard = document.querySelector(`.preset-card[data-preset="${presetId}"]`);
        if (activeCard) activeCard.classList.add('active');

        currentPresetId = presetId;
        const nameMap = {
            'sample_xray': 'sampel_infiltrat.jpg',
            'effusion': 'sampel_efusi_pleura.jpg',
            'hernia': 'sampel_hernia_diafragma.jpg',
            'pneumothorax': 'sampel_pneumotoraks.jpg',
            'normal': 'sampel_toraks_normal.jpg'
        };
        currentFileName = nameMap[presetId] || `${presetId}.jpg`;
        updatePatientDisplay();

        try {
            const res = await fetch(`/api/presets/${presetId}`);
            if (!res.ok) throw new Error('Gagal memuat preset citra');
            const blob = await res.blob();
            const reader = new FileReader();
            reader.onload = (e) => {
                currentImageBase64 = e.target.result;
                displayRadiographPreview(currentImageBase64, currentFileName);
                updateModelStatusCard('ready', 'Kasus Pasien Dimuat', `Siap menganalisis ${currentPatient.name}`);
                showToast(`Kasus klinis dimuat: ${currentPatient.name}`);
            };
            reader.readAsDataURL(blob);
        } catch (err) {
            console.error(err);
            updateModelStatusCard('error', 'Gagal Memuat Kasus', err.message);
            showToast(`Error: ${err.message}`);
        }
    }

    function updatePatientDisplay() {
        if (patientNameEl) patientNameEl.textContent = currentPatient.name;
        if (patientIdEl) patientIdEl.textContent = currentPatient.id;
        if (patientGenderEl) patientGenderEl.textContent = currentPatient.gender;
        const pBadge = document.getElementById('patient-id-badge');
        if (pBadge) pBadge.textContent = currentPatient.id;

        if (expPName) expPName.textContent = currentPatient.name;
        if (expPRm) expPRm.textContent = currentPatient.id;
        if (expPGender) expPGender.textContent = currentPatient.gender;
        if (docRefNo) docRefNo.textContent = `EXP-${currentPatient.id.replace('RM-', '')}`;
    }

    function updateModelStatusCard(status, title, desc) {
        if (!statusIconBox) return;
        statusIconBox.className = `status-icon-box status-${status}`;
        
        if (status === 'ready') {
            statusMainIcon.className = 'ri-cpu-line';
        } else if (status === 'running') {
            statusMainIcon.className = 'ri-loader-4-line spin-icon';
        } else if (status === 'success') {
            statusMainIcon.className = 'ri-checkbox-circle-fill';
        } else if (status === 'error') {
            statusMainIcon.className = 'ri-error-warning-fill';
        }

        if (modelStatusTitle) modelStatusTitle.textContent = title;
        if (modelStatusDesc) modelStatusDesc.textContent = desc;
    }

    // Default: Load sample X-ray
    loadPresetCase('sample_xray');

    // --- Diagnosis Inference & API Execution ---
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', executeDiagnosis);
    }

    async function executeDiagnosis() {
        if (isAnalyzing) return;
        if (!currentImageBase64 && !currentPresetId) {
            showToast('Silakan pilih sampel rontgen atau unggah berkas citra terlebih dahulu.');
            updateModelStatusCard('error', 'Gagal Menganalisis', 'Tidak ada citra radiografi yang dimuat.');
            return;
        }

        isAnalyzing = true;
        setAnalysisLoadingState(true);
        updateModelStatusCard('running', 'Model Sedang Memproses...', 'Mengevaluasi representasi spasial DenseNet121 & sekuensial GRU.');

        const startTime = performance.now();

        try {
            const payload = {
                threshold: currentThreshold,
                patient_name: currentPatient.name,
                patient_rm: currentPatient.id,
                patient_gender: currentPatient.gender,
                image_filename: currentFileName,
                wl_mode: currentWlMode
            };

            if (currentPresetId) {
                payload.preset_id = currentPresetId;
            } else {
                payload.image_b64 = currentImageBase64;
            }

            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Terjadi kesalahan saat memproses model AI.');
            }

            const data = await response.json();
            const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
            
            handleInferenceSuccess(data, elapsed);
            updateModelStatusCard('success', 'Model Berhasil Diterapkan', `15 patologi dievaluasi (${elapsed} detik). Cutoff: ${Math.round(currentThreshold*100)}%`);
            showToast('Evaluasi ekspertise AI & lokalisasi lesi berhasil diselesaikan.');

        } catch (err) {
            console.error(err);
            updateModelStatusCard('error', 'Model Gagal Menerapkan Analisis', err.message);
            showToast(`Gagal analisis: ${err.message}`);
        } finally {
            isAnalyzing = false;
            setAnalysisLoadingState(false);
        }
    }

    function setAnalysisLoadingState(loading) {
        if (loading) {
            analyzeBtn.disabled = true;
            analyzeIcon.className = 'ri-loader-4-line spin-icon';
            analyzeText.textContent = 'Mengevaluasi Fitur Toraks (DenseNet121+GRU)...';
        } else {
            analyzeBtn.disabled = false;
            analyzeIcon.className = 'ri-stethoscope-line';
            analyzeText.textContent = 'Jalankan Analisis AI & Score-CAM';
        }
    }

    function handleInferenceSuccess(data, elapsed = '0.15') {
        currentPredictions = data.predictions;

        // 1. Update Metrics
        metricDetected.textContent = data.total_detected;
        metricTopProb.textContent = `${data.top_probability}%`;
        metricTopName.textContent = data.top_disease;

        // 2. Structured Clean Findings Impression (No Wall of Text!)
        updateStructuredImpressionBanner(data);

        // 3. Render Disease Diagnosis Rows
        renderDiseaseList(currentPredictions);

        // 4. Populate XAI Dropdown
        populateXaiSelect(currentPredictions);

        // 5. Update Formal Structured Radiology Expertise Sheet (Tab 3)
        updateFormalExpertiseSheet(data);

        // 6. Automatically trigger Score-CAM for top finding
        const defaultXai = (data.detected_classes && data.detected_classes.length > 0) 
            ? data.detected_classes[0] 
            : data.top_disease;
        loadScoreCam(defaultXai, true);
    }

    function updateStructuredImpressionBanner(data) {
        const detected = data.detected_classes || [];
        findingsChipsRow.innerHTML = '';

        if (detected.length > 0) {
            impressionBanner.className = 'impression-banner triage-alert';
            impressionIcon.innerHTML = '<i class="ri-alert-fill"></i>';
            impressionStatusTitle.textContent = `Kesan Radiologis: ${detected.length} Patologi Terdeteksi (≥ ${Math.round(data.threshold * 100)}%)`;
            triageTag.className = 'triage-tag triage-alert';
            triageTag.textContent = 'Perhatian Klinis';

            // Render structured clean chips with quick Score-CAM trigger
            detected.forEach(diseaseName => {
                const item = data.predictions.find(p => p.class_name === diseaseName);
                const probVal = item ? item.percentage.toFixed(1) : '-';

                const chip = document.createElement('button');
                chip.type = 'button';
                chip.className = 'finding-chip';
                chip.innerHTML = `
                    <span class="chip-name">${diseaseName}</span>
                    <span class="chip-prob">${probVal}%</span>
                    <span class="chip-view-xai"><i class="ri-focus-3-line"></i> XAI</span>
                `;
                chip.addEventListener('click', () => {
                    loadScoreCam(diseaseName, true);
                    navigateToTab('tab-xai');
                    showToast(`Menampilkan peta Score-CAM untuk ${diseaseName}`);
                });
                findingsChipsRow.appendChild(chip);
            });

            if (clinicalRecNote) {
                clinicalRecNote.style.display = 'flex';
                recommendationTextContent.innerHTML = `Klik salah satu chip di atas untuk langsung melihat lokalisasi lesi pada tab <strong>Score-CAM</strong>. Konfirmasi temuan dengan dokter spesialis radiologi.`;
            }
        } else {
            impressionBanner.className = 'impression-banner triage-normal';
            impressionIcon.innerHTML = '<i class="ri-checkbox-circle-fill"></i>';
            impressionStatusTitle.textContent = 'Kesan Radiologis: Dalam Batas Normal';
            triageTag.className = 'triage-tag triage-normal';
            triageTag.textContent = 'Dalam Batas Normal';

            findingsChipsRow.innerHTML = `
                <div class="chip-normal-pill">
                    <i class="ri-shield-check-line"></i>
                    <span>Tidak ditemukan patologi di atas ambang batas ${Math.round(data.threshold * 100)}%. Probabilitas tertinggi: <strong>${data.top_disease} (${data.top_probability}%)</strong>.</span>
                </div>
            `;

            if (clinicalRecNote) {
                clinicalRecNote.style.display = 'flex';
                recommendationTextContent.innerHTML = `Semua nilai berada di bawah cutoff. Disarankan observasi klinis berkala.`;
            }
        }
    }

    function renderDiseaseList(predictions) {
        diseaseList.innerHTML = '';

        predictions.forEach((item, index) => {
            const isDet = item.probability >= currentThreshold;
            const row = document.createElement('div');
            row.className = `disease-row ${isDet ? 'status-detected' : 'status-normal'}`;

            const badgeClass = isDet ? 'badge-red' : 'badge-gray';
            const badgeText = isDet ? 'TERDETEKSI [!]' : 'DALAM BATAS NORMAL';
            const fillClass = isDet ? 'fill-red' : 'fill-gray';

            row.innerHTML = `
                <div class="row-top">
                    <div class="disease-info-left">
                        <span class="disease-rank">#${index + 1}</span>
                        <span class="disease-label">${item.class_name}</span>
                    </div>
                    <div class="disease-info-right">
                        <span class="auroc-pill" title="Benchmark AUROC Baseline">AUROC ${item.auroc.toFixed(4)}</span>
                        <span class="status-badge ${badgeClass}">${badgeText}</span>
                    </div>
                </div>
                <div class="bar-track">
                    <div class="bar-fill ${fillClass}" style="width: ${item.percentage}%;"></div>
                </div>
                <div class="row-bottom">
                    <span>Cutoff: ${Math.round(currentThreshold * 100)}%</span>
                    <span class="view-xai-cue"><i class="ri-focus-3-line"></i> Lihat Score-CAM</span>
                    <span class="prob-score">${item.percentage.toFixed(2)}%</span>
                </div>
            `;

            row.addEventListener('click', () => {
                loadScoreCam(item.class_name, true);
                navigateToTab('tab-xai');
                showToast(`Visualisasi Score-CAM untuk: ${item.class_name}`);
            });

            diseaseList.appendChild(row);
        });
    }

    function updateFormalExpertiseSheet(data) {
        const detected = data.detected_classes || [];

        const hasCardio = detected.includes('Cardiomegaly');
        const hasPulmonary = detected.some(d => ['Infiltration', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Edema', 'Emphysema', 'Fibrosis', 'Mass', 'Nodule'].includes(d));
        const hasPleural = detected.some(d => ['Effusion', 'Pneumothorax', 'Pleural_Thickening', 'Hernia'].includes(d));

        // 1. COR (Jantung & Mediastinum)
        if (hasCardio) {
            findingCor.innerHTML = 'CTR (Cardio-Thoracic Ratio) &gt; 0.50, tampak pembesaran bayangan jantung (Kardiomegali). Aorta elongasi/kalsifikasi.';
            badgeCor.innerHTML = '<span class="badge-status-alert">Kardiomegali</span>';
        } else {
            findingCor.innerHTML = 'Bentuk dan ukuran jantung dalam batas normal (CTR &le; 0.50). Mediastinum superior dan trakea di garis tengah.';
            badgeCor.innerHTML = '<span class="badge-status-ok">Normal</span>';
        }

        // 2. PULMO (Paru)
        if (hasPulmonary) {
            const pulmoDiseases = detected.filter(d => ['Infiltration', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Edema', 'Emphysema', 'Fibrosis', 'Mass', 'Nodule'].includes(d));
            findingPulmo.innerHTML = `Tampak gambaran patologi parenkim paru mencurigakan: <strong>${pulmoDiseases.join(', ')}</strong>. Corakan bronkovaskular meningkat/kasar.`;
            badgePulmo.innerHTML = '<span class="badge-status-alert">Temuan Positif</span>';
        } else {
            findingPulmo.innerHTML = 'Corakan bronkovaskular dalam batas normal. Tidak tampak infiltrat, konsolidasi, maupun nodul/massa spesifik.';
            badgePulmo.innerHTML = '<span class="badge-status-ok">Normal</span>';
        }

        // 3. PLEURA & DIAFRAGMA
        if (hasPleural) {
            const pleuraDiseases = detected.filter(d => ['Effusion', 'Pneumothorax', 'Pleural_Thickening', 'Hernia'].includes(d));
            findingPleura.innerHTML = `Tampak tanda kelainan pleura/diafragma: <strong>${pleuraDiseases.join(', ')}</strong>. Sudut kostofrenikus atau kubah diafragma tidak intak.`;
            badgePleura.innerHTML = '<span class="badge-status-alert">Temuan Positif</span>';
        } else {
            findingPleura.innerHTML = 'Sudut kostofrenikus kanan dan kiri lancip tajam. Kedua kubah diafragma licin dan simetris.';
            badgePleura.innerHTML = '<span class="badge-status-ok">Normal</span>';
        }

        // 4. SKELETAL & SOFT TISSUE
        findingBones.innerHTML = 'Struktur tulang costae, clavicula, dan vertebrae intak. Jaringan lunak dinding dada simetris tenang.';
        badgeBones.innerHTML = '<span class="badge-status-ok">Normal</span>';

        // 5. CONCLUSION / IMPRESSION
        if (detected.length > 0) {
            const itemsList = detected.map((d, i) => {
                const p = data.predictions.find(item => item.class_name === d);
                return `<p>${i + 1}. <strong>${d}</strong> (Probabilitas Model AI: <strong>${p ? p.percentage : '-'}%</strong>).</p>`;
            }).join('');

            expertiseConclusion.innerHTML = `
                ${itemsList}
                <p class="recommendation-text"><em>Saran Medis: Konfirmasi korelasi dengan klinis pasien, laboratorium darah rutin, dan follow-up evaluasi radiologi.</em></p>
            `;
        } else {
            expertiseConclusion.innerHTML = `
                <p>1. <strong>Foto Toraks dalam batas normal radiologis (No Finding)</strong>.</p>
                <p>2. Tidak tampak kardiomegali, infiltrat pulmo, maupun efusi pleura bermakna.</p>
            `;
        }
    }

    function reEvaluateDiagnosis() {
        let detectedList = [];
        currentPredictions.forEach(p => {
            p.is_detected = p.probability >= currentThreshold;
            if (p.is_detected) detectedList.push(p.class_name);
        });

        currentPredictions.sort((a, b) => {
            if (a.is_detected !== b.is_detected) return b.is_detected - a.is_detected;
            return b.probability - a.probability;
        });

        metricDetected.textContent = detectedList.length;

        const topPred = currentPredictions[0];
        const updatePayload = {
            threshold: currentThreshold,
            detected_classes: detectedList,
            predictions: currentPredictions,
            top_disease: topPred ? topPred.class_name : '-',
            top_probability: topPred ? topPred.percentage : 0
        };

        updateStructuredImpressionBanner(updatePayload);
        renderDiseaseList(currentPredictions);
        populateXaiSelect(currentPredictions);
        updateFormalExpertiseSheet(updatePayload);
    }

    function populateXaiSelect(predictions) {
        xaiDiseaseSelect.innerHTML = '';
        predictions.forEach(item => {
            const isDet = item.probability >= currentThreshold;
            const opt = document.createElement('option');
            opt.value = item.class_name;
            opt.textContent = `${item.class_name} (${item.percentage.toFixed(1)}%) ${isDet ? '★ Terdeteksi' : ''}`;
            xaiDiseaseSelect.appendChild(opt);
        });

        if (currentXaiDisease) {
            xaiDiseaseSelect.value = currentXaiDisease;
        }
    }

    if (xaiDiseaseSelect) {
        xaiDiseaseSelect.addEventListener('change', (e) => {
            loadScoreCam(e.target.value, false);
        });
    }

    // --- Score-CAM Explainability Engine (with Real-time Layered Overlay) ---
    async function loadScoreCam(diseaseName, syncSelect = true) {
        currentXaiDisease = diseaseName;
        if (syncSelect && xaiDiseaseSelect) {
            xaiDiseaseSelect.value = diseaseName;
        }

        const predObj = currentPredictions.find(p => p.class_name === diseaseName);
        const probText = predObj ? `${predObj.percentage.toFixed(2)}%` : '-';
        const aurocText = predObj ? predObj.auroc.toFixed(4) : '-';

        xaiTagProb.textContent = `Prob: ${probText}`;
        xaiTagAuroc.textContent = `AUROC: ${aurocText}`;

        loadOrig.style.display = 'none';
        loadHeat.style.display = 'none';
        loadOver.style.display = 'none';

        try {
            const payload = {
                class_name: diseaseName,
                alpha: currentAlpha,
                top_k: 20
            };

            if (currentPresetId) {
                payload.preset_id = currentPresetId;
            } else if (currentImageBase64) {
                payload.image_b64 = currentImageBase64;
            }

            const res = await fetch('/api/scorecam', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error('Gagal menghasilkan Score-CAM');
            const data = await res.json();

            // Panel 1: Original X-Ray
            showcaseImgOrig.src = data.original_image;
            showcaseImgOrig.style.display = 'block';

            // Panel 2: Heatmap
            showcaseImgHeat.src = data.heatmap_image;
            showcaseImgHeat.style.display = 'block';

            // Panel 3: Instant Layered Interactive Overlay
            overlayBaseImg.src = data.original_image;
            overlayBaseImg.style.display = 'block';

            overlayHeatImg.src = data.heatmap_image;
            overlayHeatImg.style.display = 'block';
            overlayHeatImg.style.opacity = currentAlpha; // Direct real-time alpha

            // Natural Clinical Narrative for XAI
            const isPos = data.probability >= currentThreshold;
            if (isPos) {
                interpretationDesc.innerHTML = `
                    <strong>Aktivasi Dominan untuk ${diseaseName} (${data.percentage}%):</strong> Algoritma Score-CAM mengidentifikasi bahwa representasi fitur konvolusi DenseNet121 terkonsentrasi kuat pada regio toraks bersangkutan. Area berwarna merah/panas memberikan bobot keyakinan terbesar bagi model GRU untuk mendeteksi <em>${diseaseName}</em>.
                `;
            } else {
                interpretationDesc.innerHTML = `
                    <strong>Aktivasi Rendah untuk ${diseaseName} (${data.percentage}%):</strong> Nilai probabilitas berada di bawah cutoff ambang klinis (${Math.round(currentThreshold*100)}%). Heatmap mengilustrasikan distribusi fitur laten yang dievaluasi model sebelum mengecualikan patologi ini.
                `;
            }

        } catch (err) {
            console.error(err);
            showToast(`Error visualisasi Score-CAM: ${err.message}`);
        }
    }

    // --- Print / Export Radiology Expertise Sheet ---
    if (printReportBtn) {
        printReportBtn.addEventListener('click', () => {
            navigateToTab('tab-report');
            setTimeout(() => {
                window.print();
            }, 300);
        });
    }

    // --- PACS Database & Cloud History Modal Controller ---
    const openDbBtn = document.getElementById('open-db-history-btn');
    const closeDbBtn = document.getElementById('close-db-history-btn');
    const closeDbFooterBtn = document.getElementById('close-db-history-footer-btn');
    const dbModal = document.getElementById('db-history-modal');
    const refreshDbBtn = document.getElementById('refresh-db-btn');
    const dbTbody = document.getElementById('db-history-tbody');
    const dbTotalRecords = document.getElementById('db-total-records');
    const dbSearchInput = document.getElementById('db-search-input');
    const dbStatusPill = document.getElementById('db-status-pill');
    const dbConnText = document.getElementById('db-conn-text');

    let allDbRecords = [];

    // Check DB status on startup
    checkDatabaseHealth();

    async function checkDatabaseHealth() {
        try {
            const res = await fetch('/api/db-health');
            if (!res.ok) throw new Error('DB Offline');
            const data = await res.json();
            if (data.supabase_connected) {
                if (dbStatusPill) {
                    dbStatusPill.textContent = 'Cloud ON';
                    dbStatusPill.style.color = '#22c55e';
                }
                if (dbConnText) dbConnText.textContent = 'Terhubung (Online)';
            } else {
                if (dbStatusPill) {
                    dbStatusPill.textContent = 'Cloud OFF';
                    dbStatusPill.style.color = '#eab308';
                }
                if (dbConnText) dbConnText.textContent = 'Menunggu Konfigurasi';
            }
        } catch (e) {
            if (dbStatusPill) {
                dbStatusPill.textContent = 'Offline';
                dbStatusPill.style.color = '#ef4444';
            }
        }
    }

    if (openDbBtn) {
        openDbBtn.addEventListener('click', () => {
            if (dbModal) dbModal.style.display = 'flex';
            fetchDatabaseHistory();
        });
    }

    if (closeDbBtn) {
        closeDbBtn.addEventListener('click', () => {
            if (dbModal) dbModal.style.display = 'none';
        });
    }

    if (closeDbFooterBtn) {
        closeDbFooterBtn.addEventListener('click', () => {
            if (dbModal) dbModal.style.display = 'none';
        });
    }

    // Close on clicking outside modal card
    if (dbModal) {
        dbModal.addEventListener('click', (e) => {
            if (e.target === dbModal) {
                dbModal.style.display = 'none';
            }
        });
    }

    if (refreshDbBtn) {
        refreshDbBtn.addEventListener('click', fetchDatabaseHistory);
    }

    if (dbSearchInput) {
        dbSearchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            filterAndRenderHistory(query);
        });
    }

    async function fetchDatabaseHistory() {
        if (!dbTbody) return;
        dbTbody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center py-4 text-muted">
                    <i class="ri-loader-4-line spin-icon"></i> Memuat data riwayat dari Supabase PostgreSQL...
                </td>
            </tr>
        `;

        try {
            const res = await fetch('/api/history?limit=50');
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.error || 'Gagal memuat riwayat');
            }
            const data = await res.json();
            allDbRecords = data.history || [];
            if (dbTotalRecords) dbTotalRecords.textContent = `${allDbRecords.length} Pemeriksaan`;
            filterAndRenderHistory(dbSearchInput ? dbSearchInput.value.toLowerCase().trim() : '');
        } catch (err) {
            console.error(err);
            dbTbody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center py-4 text-rose">
                        <i class="ri-error-warning-line"></i> Gagal mengambil data: ${err.message}
                    </td>
                </tr>
            `;
        }
    }

    function filterAndRenderHistory(query = '') {
        if (!dbTbody) return;

        let filtered = allDbRecords;
        if (query) {
            filtered = allDbRecords.filter(r => {
                const name = (r.full_name || '').toLowerCase();
                const rm = (r.rm_number || '').toLowerCase();
                const code = (r.study_code || '').toLowerCase();
                const top = (r.top_disease || '').toLowerCase();
                return name.includes(query) || rm.includes(query) || code.includes(query) || top.includes(query);
            });
        }

        if (filtered.length === 0) {
            dbTbody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center py-4 text-muted">
                        <i class="ri-inbox-line"></i> ${query ? 'Tidak ada data yang cocok dengan pencarian.' : 'Belum ada data diagnosa tersimpan.'}
                    </td>
                </tr>
            `;
            return;
        }

        dbTbody.innerHTML = filtered.map(row => {
            const timeStr = row.diagnosis_time ? new Date(row.diagnosis_time).toLocaleString('id-ID', {
                day: '2-digit', month: 'short', year: 'numeric',
                hour: '2-digit', minute: '2-digit'
            }) : '-';

            const isAlert = row.triage_status === 'alert' || (row.total_detected > 0);
            const badgeClass = isAlert ? 'db-badge-alert' : 'db-badge-normal';
            const badgeText = isAlert ? `${row.total_detected || 1} PATOLOGI` : 'NORMAL';

            return `
                <tr>
                    <td>
                        <div class="font-mono font-bold text-xs">${row.study_code || '-'}</div>
                        <div class="text-xs text-muted">${timeStr}</div>
                    </td>
                    <td>
                        <div class="font-bold">${escapeHtml(row.full_name || 'Pasien Anonim')}</div>
                        <div class="text-xs font-mono text-muted">${escapeHtml(row.rm_number || '-')} • ${row.age ? row.age + ' thn' : ''} ${row.gender || ''}</div>
                    </td>
                    <td>
                        <span class="text-xs">${row.projection || 'AP/PA'}</span>
                    </td>
                    <td>
                        <strong class="text-xs text-emerald">${escapeHtml(row.top_disease || '-')}</strong>
                    </td>
                    <td>
                        <span class="font-mono font-bold text-xs">${row.top_probability ? row.top_probability + '%' : '-'}</span>
                    </td>
                    <td>
                        <span class="db-badge-triage ${badgeClass}">
                            <i class="${isAlert ? 'ri-alarm-warning-line' : 'ri-check-line'}"></i>
                            ${badgeText}
                        </span>
                    </td>
                    <td>
                        <button class="btn btn-sm btn-subtle load-history-case-btn" data-preset="${row.preset_id || 'sample_xray'}" data-name="${escapeHtml(row.full_name || '')}" data-rm="${escapeHtml(row.rm_number || '')}" data-gender="${escapeHtml(row.gender || '')}" title="Muat kasus ini ke viewer">
                            <i class="ri-folder-open-line"></i> Buka
                        </button>
                    </td>
                </tr>
            `;
        }).join('');

        // Attach load buttons
        document.querySelectorAll('.load-history-case-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const pPreset = btn.getAttribute('data-preset');
                const pName = btn.getAttribute('data-name');
                const pRm = btn.getAttribute('data-rm');
                const pGender = btn.getAttribute('data-gender');

                currentPatient = {
                    name: pName || 'Pasien Anonim',
                    id: pRm || 'RM-2026-00000',
                    gender: pGender || 'Laki-laki'
                };

                if (dbModal) dbModal.style.display = 'none';
                loadPresetCase(pPreset || 'sample_xray');
                showToast(`Kasus dibuka dari Database: ${currentPatient.name} (${currentPatient.id})`);
            });
        });
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str).replace(/[&<>"']/g, function (m) {
            return {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;'
            }[m];
        });
    }

    // --- Toast Controller ---
    function showToast(msg) {
        if (!appToast) return;
        appToast.textContent = msg;
        appToast.style.display = 'block';

        if (toastTimer) clearTimeout(toastTimer);
        toastTimer = setTimeout(() => {
            appToast.style.display = 'none';
        }, 3000);
    }
});

