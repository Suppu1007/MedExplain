/* =====================================================
   TOOLTIP INITIALIZATION (Bootstrap)
===================================================== */
document.querySelectorAll('[data-bs-toggle="tooltip"]')
  .forEach(el => new bootstrap.Tooltip(el));


/* =====================================================
   PASSWORD VISIBILITY TOGGLE
===================================================== */
function togglePass(inputId, iconEl) {
  const input = document.getElementById(inputId);
  if (!input) return;

  const isHidden = input.type === "password";
  input.type = isHidden ? "text" : "password";
  iconEl.textContent = isHidden ? "visibility_off" : "visibility";
}


/* =====================================================
   PASSWORD STRENGTH + MATCH VALIDATION
===================================================== */
const passwordInput = document.getElementById("password");
const confirmInput = document.getElementById("confirm");
const signupBtn = document.getElementById("signupBtn");

let captchaVerified = false;

/**
 * Password rules:
 * - Starts with uppercase letter
 * - Minimum length: 7
 * - At least one special character
 * - Must match confirmation field
 */
function validateForm() {
  if (!passwordInput || !confirmInput || !signupBtn) return;

  const password = passwordInput.value;
  const confirm = confirmInput.value;

  const strongPassword =
    /^[A-Z]/.test(password) &&
    password.length >= 7 &&
    /[^A-Za-z0-9]/.test(password);

  const passwordsMatch = password === confirm;

  signupBtn.disabled = !(
    strongPassword &&
    passwordsMatch &&
    captchaVerified
  );
}


/* =====================================================
   INPUT LISTENERS
===================================================== */
passwordInput?.addEventListener("input", validateForm);
confirmInput?.addEventListener("input", validateForm);


/* =====================================================
   GOOGLE reCAPTCHA CALLBACK
===================================================== */
function onCaptchaSuccess() {
  captchaVerified = true;
  validateForm();
}
